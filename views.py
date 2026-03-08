import logging
import os
import uuid
import csv
import io
import json
import hashlib
import html
from datetime import datetime

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.urls import reverse
from django.http import HttpResponse, FileResponse, JsonResponse
from django.views.decorators.http import require_http_methods

from .forms import UploadCSVForm
from .predictor import analyze_csv

logger = logging.getLogger(__name__)


def _normalize_jsonable(value):
    """Recursively convert numpy/pandas scalars and complex objects to JSON-safe types."""
    if isinstance(value, dict):
        return {k: _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(v) for v in value]
    if hasattr(value, 'item') and callable(getattr(value, 'item')):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


def _build_chart_context(result):
    """Build chart payload used by the dashboard template."""
    if not result:
        return {}

    chart_context = {}
    if 'history' in result:
        chart_context = {
            'dates': json.dumps(result['history'].get('dates', [])),
            'sales': json.dumps(result['history'].get('sales', [])),
            'forecast': json.dumps(result.get('predicted_demand', 0))
        }
        if 'currency' in result:
            chart_context['currency'] = result['currency']

    if 'bar_chart' in result:
        chart_context['bar'] = result.get('bar_chart', {})
    if 'pie_chart' in result:
        chart_context['pie'] = result.get('pie_chart', {})

    return chart_context


def _build_chart_payload(result):
    """JSON-safe chart payload used by frontend bootstrap."""
    if not result:
        return {}

    return _normalize_jsonable({
        'dates': result.get('history', {}).get('dates', []),
        'sales': result.get('history', {}).get('sales', []),
        'forecast': float(result.get('predicted_demand', 0) or 0),
        'bar_labels': result.get('bar_chart', {}).get('labels', []),
        'bar_values': result.get('bar_chart', {}).get('values', []),
        'pie_labels': result.get('pie_chart', {}).get('labels', []),
        'pie_values': result.get('pie_chart', {}).get('values', []),
    })


def _build_server_chart_svg(result):
    """Build a deterministic SVG chart so chart area is never blank."""
    if not result:
        return ""

    history = result.get('history', {})
    dates = history.get('dates', []) or []
    sales = [float(v) for v in (history.get('sales', []) or [])]
    forecast = float(result.get('predicted_demand') or 0.0)
    bar_labels = result.get('bar_chart', {}).get('labels', []) or []
    bar_values = [float(v) for v in (result.get('bar_chart', {}).get('values', []) or [])]
    pie_labels = result.get('pie_chart', {}).get('labels', []) or []
    pie_values = [float(v) for v in (result.get('pie_chart', {}).get('values', []) or [])]

    width, height = 900, 320
    ml, mr, mt, mb = 60, 20, 20, 45
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    def esc(s):
        return html.escape(str(s))

    if dates and sales:
        labels = list(dates) + ["Forecast"]
        values = list(sales) + [forecast]
        y_max = max(1.0, max(values))
        n = len(labels)
        group_w = plot_w / max(1, n)
        bar_w = max(10, group_w * 0.55)

        bars = []
        ticks = []
        for i, label in enumerate(labels):
            gx = ml + i * group_w
            x = gx + (group_w - bar_w) / 2
            y = mt + plot_h - (float(values[i]) / y_max) * plot_h
            fill = "#2563eb" if i < len(sales) else "#9333ea"
            opacity = "0.78" if i < len(sales) else "0.85"
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{(mt+plot_h-y):.1f}" '
                f'rx="5" ry="5" fill="{fill}" opacity="{opacity}"/>'
            )
            if i % max(1, n // 6) == 0 or i == n - 1:
                ticks.append(
                    f'<text x="{gx + group_w/2:.1f}" y="{height-18}" text-anchor="middle" '
                    f'font-size="10" fill="#6b7280">{esc(label)}</text>'
                )

        return (
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="100%" preserveAspectRatio="none">'
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#e5e7eb"/>'
            f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#e5e7eb"/>'
            f'{"".join(bars)}{"".join(ticks)}'
            f'</svg>'
        )

    # Fallback to top-product bars when no timeline exists.
    labels = list(dict.fromkeys(list(bar_labels) + list(pie_labels)))
    if not labels:
        return ""

    sold_map = {k: float(v) for k, v in zip(bar_labels, bar_values)}
    rec_map = {k: float(v) for k, v in zip(pie_labels, pie_values)}
    sold = [sold_map.get(k, 0.0) for k in labels]
    recs = [rec_map.get(k, 0.0) for k in labels]
    y_max = max(1.0, max(sold + recs))
    n = len(labels)
    group_w = plot_w / max(1, n)
    bar_w = max(8, group_w * 0.28)

    bars = []
    ticks = []
    for i, name in enumerate(labels):
        gx = ml + i * group_w
        sx = gx + (group_w - (2 * bar_w + 4)) / 2
        rx = sx + bar_w + 4
        sy = mt + plot_h - (sold[i] / y_max) * plot_h
        ry = mt + plot_h - (recs[i] / y_max) * plot_h
        bars.append(f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{bar_w:.1f}" height="{(mt+plot_h-sy):.1f}" fill="#3b82f6" opacity="0.75"/>')
        bars.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{bar_w:.1f}" height="{(mt+plot_h-ry):.1f}" fill="#10b981" opacity="0.70"/>')
        short = esc(name[:12] + (".." if len(name) > 12 else ""))
        ticks.append(f'<text x="{gx+group_w/2:.1f}" y="{height-18}" text-anchor="middle" font-size="10" fill="#6b7280">{short}</text>')

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="100%" preserveAspectRatio="none">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#e5e7eb"/>'
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#e5e7eb"/>'
        f'{"".join(bars)}{"".join(ticks)}'
        f'</svg>'
    )


def _build_server_pie_svg(result):
    """Build server-side doughnut chart fallback SVG."""
    if not result:
        return ""

    pie_labels = result.get('pie_chart', {}).get('labels', []) or []
    pie_values = [float(v) for v in (result.get('pie_chart', {}).get('values', []) or [])]
    history_sales = [float(v) for v in (result.get('history', {}).get('sales', []) or [])]
    forecast = float(result.get('predicted_demand') or 0.0)

    labels = list(pie_labels)
    values = list(pie_values)
    if not labels or not values:
        hist_total = sum(history_sales)
        labels = ["Historical Total", "Forecast"]
        values = [hist_total, forecast]

    total = sum(values)
    if total <= 0:
        return ""

    colors = ["#2563eb", "#9333ea", "#10b981", "#f59e0b", "#ef4444", "#14b8a6", "#0ea5e9"]
    cx, cy, r_outer, r_inner = 185, 160, 108, 56
    start = -90.0
    arcs = []
    legends = []

    for i, val in enumerate(values[:7]):
        fraction = (float(val) / total) if total else 0
        sweep = max(0.0, fraction * 360.0)
        end = start + sweep
        color = colors[i % len(colors)]
        x1 = cx + r_outer * __import__('math').cos(__import__('math').radians(start))
        y1 = cy + r_outer * __import__('math').sin(__import__('math').radians(start))
        x2 = cx + r_outer * __import__('math').cos(__import__('math').radians(end))
        y2 = cy + r_outer * __import__('math').sin(__import__('math').radians(end))
        x3 = cx + r_inner * __import__('math').cos(__import__('math').radians(end))
        y3 = cy + r_inner * __import__('math').sin(__import__('math').radians(end))
        x4 = cx + r_inner * __import__('math').cos(__import__('math').radians(start))
        y4 = cy + r_inner * __import__('math').sin(__import__('math').radians(start))
        large = 1 if sweep > 180 else 0
        path = (
            f'M {x1:.2f},{y1:.2f} A {r_outer},{r_outer} 0 {large},1 {x2:.2f},{y2:.2f} '
            f'L {x3:.2f},{y3:.2f} A {r_inner},{r_inner} 0 {large},0 {x4:.2f},{y4:.2f} Z'
        )
        arcs.append(f'<path d="{path}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        pct = (float(val) / total) * 100.0 if total else 0.0
        label = html.escape(str(labels[i])[:18] + (".." if len(str(labels[i])) > 18 else ""))
        legends.append(
            f'<rect x="368" y="{36 + i*36}" width="12" height="12" rx="2" ry="2" fill="{color}"/>'
            f'<text x="386" y="{47 + i*36}" font-size="11" fill="#374151">{label}</text>'
            f'<text x="386" y="{60 + i*36}" font-size="10" fill="#6b7280">{val:.0f} ({pct:.1f}%)</text>'
        )
        start = end

    return (
        '<svg viewBox="0 0 620 320" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">'
        '<rect x="0" y="0" width="620" height="320" fill="#ffffff"/>'
        + "".join(arcs)
        + '<circle cx="185" cy="160" r="46" fill="#f8fafc"/>'
        + '<text x="185" y="154" text-anchor="middle" font-size="11" fill="#6b7280">Total</text>'
        + f'<text x="185" y="171" text-anchor="middle" font-size="14" font-weight="700" fill="#111827">{total:.0f}</text>'
        + "".join(legends)
        + '</svg>'
    )


def index(request):
    """Display upload form and handle CSV upload (saves to MEDIA_ROOT/uploads/)."""
    if request.method == 'POST':
        if 'file' not in request.FILES or not request.FILES.get('file'):
            messages.error(request, 'No file uploaded. Please choose a CSV file.')
            return redirect('analytics:index')

        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['file']
            
            # Validation
            filename = getattr(csv_file, 'name', '')
            if not filename.lower().endswith('.csv'):
                messages.error(request, 'Uploaded file is not a CSV. Please upload a .csv file.')
                return redirect('analytics:index')

            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            unique_name = f"uploads/{uuid.uuid4().hex}_{filename}"
            
            try:
                saved_name = fs.save(unique_name, csv_file)
                logger.info(f"File uploaded successfully: {saved_name}")
            except Exception as e:
                logger.exception(f"Failed to save file: {e}")
                messages.error(request, f'Failed to save uploaded file: {e}')
                return redirect('analytics:index')

            messages.success(request, 'File uploaded successfully.')
            return redirect(f"{reverse('analytics:process')}?file={saved_name}")
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = UploadCSVForm()

    return render(request, 'analytics/index.html', {'form': form})


def dashboard(request):
    """
    Redirect Dashboard nav clicks to the chart section of the latest (or requested) analysis.
    Priority:
    1) explicit `file` query param
    2) latest uploaded CSV under MEDIA_ROOT/uploads
    """
    file_rel = request.GET.get('file')
    if file_rel:
        file_path = os.path.join(settings.MEDIA_ROOT, file_rel)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return redirect(f"{reverse('analytics:process')}?file={file_rel}#charts-section")

    uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    latest_rel = None
    try:
        if os.path.isdir(uploads_dir):
            csv_files = []
            for name in os.listdir(uploads_dir):
                if not name.lower().endswith('.csv'):
                    continue
                abs_path = os.path.join(uploads_dir, name)
                if os.path.isfile(abs_path):
                    csv_files.append((os.path.getmtime(abs_path), name))
            if csv_files:
                csv_files.sort(key=lambda x: x[0], reverse=True)
                latest_rel = f"uploads/{csv_files[0][1]}"
    except Exception as e:
        logger.warning(f"Could not resolve latest dashboard file: {e}")

    if latest_rel:
        return redirect(f"{reverse('analytics:process')}?file={latest_rel}#charts-section")

    messages.info(request, 'Upload a CSV first to view the dashboard charts.')
    return redirect('analytics:index')


def process(request):
    """Simple processing page — receives a `file` query param pointing to path under MEDIA_ROOT."""
    file_rel = request.GET.get('file')
    context = {}
    if file_rel:
        file_path = os.path.join(settings.MEDIA_ROOT, file_rel)
        context['uploaded_file'] = file_rel
        context['exists'] = os.path.exists(file_path)
        
        if context['exists']:
            try:
                logger.info(f"Analyzing file: {file_path}")
                result = analyze_csv(file_path)
                return render(request, 'analytics/results.html', {
                    'result': result,
                    'uploaded_file': file_rel,
                    'chart_data': _build_chart_context(result),
                    'chart_payload_json': _build_chart_payload(result),
                    'chart_svg': _build_server_chart_svg(result),
                    'chart_pie_svg': _build_server_pie_svg(result),
                })
            except ValueError as e:
                logger.warning(f"Validation error during analysis: {e}")
                messages.error(request, f'Analysis Error: {e}')
            except Exception as e:
                logger.exception(f"Unexpected analysis error: {e}")
                messages.error(request, f'An unexpected error occurred: {e}')
    
    return render(request, 'analytics/process.html', context)


def results(request):
    """Run prediction on the uploaded file and render results page."""
    file_rel = request.GET.get('file')
    if not file_rel:
        messages.error(request, 'No file specified for prediction.')
        return redirect('analytics:index')

    file_path = os.path.join(settings.MEDIA_ROOT, file_rel)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        messages.error(request, 'Uploaded file not found or is invalid.')
        return redirect('analytics:index')

    try:
        result = analyze_csv(file_path)
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        messages.error(request, f'Error during prediction: {e}')
        # Return to process page or show results with error state
        return render(request, 'analytics/results.html', {'result': None, 'uploaded_file': file_rel})
    
    # Save output CSV logic remains similar, maybe extract to helper?
    output_rel = _save_prediction_csv(result)
    
    chart_context = _build_chart_context(result)

    return render(request, 'analytics/results.html', {
        'result': result, 
        'uploaded_file': file_rel, 
        'output_file': output_rel,
        'chart_data': chart_context,
        'chart_payload_json': _build_chart_payload(result),
        'chart_svg': _build_server_chart_svg(result),
        'chart_pie_svg': _build_server_pie_svg(result),
    })


def download_results(request):
    """Generate a CSV download of prediction results for the given uploaded file."""
    file_rel = request.GET.get('file')
    if not file_rel:
        messages.error(request, 'No file specified for download.')
        return redirect('analytics:index')

    file_path = os.path.join(settings.MEDIA_ROOT, file_rel)
    if not os.path.exists(file_path):
        messages.error(request, 'Uploaded file not found.')
        return redirect('analytics:index')

    try:
        result = analyze_csv(file_path)
    except Exception as e:
        messages.error(request, f'Error generating download: {e}')
        return redirect('analytics:process')

    return _generate_csv_response(result)


@require_http_methods(["GET"])
def api_chart_data(request):
    """API endpoint that returns chart data in JSON format for real-time updates."""
    file_rel = request.GET.get('file')
    if not file_rel:
        return JsonResponse({'error': 'No file specified'}, status=400)

    file_path = os.path.join(settings.MEDIA_ROOT, file_rel)
    if not os.path.exists(file_path):
        return JsonResponse({'error': 'File not found'}, status=404)

    try:
        result = analyze_csv(file_path)
        predicted_demand = float(result.get('predicted_demand') or 0.0)
        est_revenue_raw = result.get('estimated_revenue')
        estimated_revenue = float(est_revenue_raw) if est_revenue_raw is not None else 0.0
        suggested_stock = int(result.get('suggested_stock') or 0)
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(timespec='seconds')

        signature_payload = {
            'predicted_demand': predicted_demand,
            'suggested_stock': suggested_stock,
            'history': result.get('history', {}),
            'top_products': result.get('top_products', []),
            'restock_recs': result.get('restock_recs', []),
            'bar_chart': result.get('bar_chart', {}),
            'pie_chart': result.get('pie_chart', {}),
        }
        data_signature = hashlib.sha256(
            json.dumps(_normalize_jsonable(signature_payload), sort_keys=True, default=str).encode('utf-8')
        ).hexdigest()
        
        # Prepare chart data
        chart_data = _normalize_jsonable({
            'success': True,
            'generated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'file_last_modified': file_mtime,
            'data_signature': data_signature,
            'predicted_demand': predicted_demand,
            'estimated_revenue': estimated_revenue,
            'suggested_stock': suggested_stock,
            'dates': result.get('history', {}).get('dates', []),
            'sales': result.get('history', {}).get('sales', []),
            'forecast': predicted_demand,
            'top_products': result.get('top_products', []),
            'restock_recs': result.get('restock_recs', []),
            'bar_labels': result.get('bar_chart', {}).get('labels', []),
            'bar_values': result.get('bar_chart', {}).get('values', []),
            'pie_labels': result.get('pie_chart', {}).get('labels', []),
            'pie_values': result.get('pie_chart', {}).get('values', []),
            'report': result.get('report', ''),
            'model_info': result.get('model_info', {}),
            'currency': result.get('currency', '$')
        })
        return JsonResponse(chart_data)
    except Exception as e:
        logger.exception(f"API error while processing chart data: {e}")
        return JsonResponse({'error': str(e)}, status=500)



def download_output(request):
    """Stream an existing output CSV."""
    file_rel = request.GET.get('out')
    if not file_rel:
        messages.error(request, 'No output file specified.')
        return redirect('analytics:index')
    
    # Security check for path traversal
    safe_rel = os.path.normpath(file_rel).replace('..', '')
    if safe_rel.startswith('/') or safe_rel.startswith('\\'):
         safe_rel = safe_rel.lstrip('/\\')

    out_path = os.path.join(settings.MEDIA_ROOT, safe_rel)
    if not os.path.exists(out_path):
        messages.error(request, 'Requested output file not found.')
        return redirect('analytics:results')

    return FileResponse(open(out_path, 'rb'), as_attachment=True, filename=os.path.basename(out_path))


def _save_prediction_csv(result):
    """Refactored helper to save prediction results to CSV."""
    try:
        outputs_dir = os.path.join(settings.MEDIA_ROOT, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        out_name = f"prediction_{uuid.uuid4().hex}.csv"
        out_path = os.path.join(outputs_dir, out_name)

        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['predicted_demand', result.get('predicted_demand')])
            writer.writerow(['estimated_revenue', result.get('estimated_revenue') or ''])
            writer.writerow(['suggested_stock', result.get('suggested_stock')])
            writer.writerow(['model', result.get('model_info', {}).get('model_used')])
            writer.writerow(['mae', result.get('model_info', {}).get('mae')])
            writer.writerow(['report', result.get('report', '').replace('\n', ' | ')])
        
        return f"outputs/{out_name}"
    except Exception as e:
        logger.exception(f"Failed to save prediction CSV: {e}")
        return None

def _generate_csv_response(result):
    """Helper to generate CSV response."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['metric', 'value'])
    writer.writerow(['predicted_demand', result.get('predicted_demand')])
    writer.writerow(['estimated_revenue', result.get('estimated_revenue') or ''])
    writer.writerow(['suggested_stock', result.get('suggested_stock')])
    writer.writerow(['model', result.get('model_info', {}).get('model_used')])
    writer.writerow(['mae', result.get('model_info', {}).get('mae')])
    writer.writerow(['report', result.get('report', '').replace('\n', ' | ')])

    csv_content = buf.getvalue()
    buf.close()

    response = HttpResponse(csv_content, content_type='text/csv')
    fname = f"prediction_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
    response['Content-Disposition'] = f'attachment; filename="{fname}"'
    return response


# End of views.py
