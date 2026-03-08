"""Prediction utilities for the `analytics` app.

This module provides a robust DemandPredictor class to handle:
- CSV loading and cleaning
- Date parsing with multiple fallback strategies
- Feature engineering (lags, rolling averages)
- Model training and prediction (RandomForest/LinearRegression)
- Business reporting and visualization data preparation
"""

import logging
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import openai

# Configure logging
logger = logging.getLogger(__name__)


class DemandPredictor:
    """
    Encapsulates the logic for loading sales data, training a predictive model,
    and generating business insights.
    """

    def __init__(self, 
                 n_lags: int = 3, 
                 lead_time_months: float = 1.0, 
                 service_level_z: float = 1.65):
        """
        Args:
            n_lags: Number of months of lag to use as features.
            lead_time_months: Assumed lead time for stock replenishment.
            service_level_z: Z-score for safety stock calculation (1.65 ~= 95% service level).
        """
        self.n_lags = n_lags
        self.lead_time_months = lead_time_months
        self.service_level_z = service_level_z
        self.model = None
        self.mae = None
        self.rmse = None
        # simple in-memory cache keyed by (path, mtime) -> analysis result
        # useful when the same CSV is analyzed repeatedly during a session
        self._cache: Dict[Tuple[str, float], Dict[str, Any]] = {}

    def analyze(self, 
                csv_path: Union[str, Path], 
                date_col: Optional[str] = None, 
                sales_col: Optional[str] = None, 
                price_col: Optional[str] = None,
                item_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point to analyze a CSV file.
        Now supports item-level analysis if an item column is present.
        
        Returns:
             Dict containing prediction results and reports.
        """
        # attempt to return cached result if available
        try:
            path_str = str(csv_path)
            try:
                mtime = os.path.getmtime(path_str)
            except OSError:
                mtime = None
            cache_key = (path_str, mtime)
            if cache_key in self._cache:
                logger.debug(f"Using cached analysis for {path_str}")
                return self._cache[cache_key]

            df, item_col_name, currency = self._load_and_validate_data(csv_path, date_col, sales_col, price_col, item_col)

            if df.empty:
                 raise ValueError("The dataset is empty after validation.")

            if item_col_name:
                result = self._analyze_per_item(df, item_col_name)
            else:
                result = self._analyze_aggregate(df)
            # stamp currency
            result['currency'] = currency

            # store in cache if modification time available
            if mtime is not None:
                self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            # Re-raise as ValueError with user-friendly message
            raise ValueError(f"Analysis failed: {str(e)}")

    def _analyze_aggregate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs the original aggregate analysis logic."""
        # Aggregate to monthly level
        monthly_df = self._aggregate_monthly(df)
        
        if len(monthly_df) < self.n_lags + 2:
            logger.warning("Insufficient data for ML model. Using simple heuristics.")
            return self._simple_heuristic_analysis(monthly_df, df, 'price' if 'price' in df else None)

        # Feature Engineering & Training
        features_df = self._create_features(monthly_df)
        X = features_df.drop(columns=['sales', 'date']).values
        y = features_df['sales'].values
        
        self._train_model(X, y)
        
        # Predict
        last_known_values = monthly_df['sales'].iloc[-self.n_lags:].values[::-1]
        next_month_pred = self._predict_single(last_known_values)

        # Build basic chart payloads for aggregate results
        bar_chart = {
            'labels': monthly_df['date'].dt.strftime('%Y-%m').tolist(),
            'values': monthly_df['sales'].tolist()
        }
        pie_chart = {
            'labels': ['Historical Total', 'Forecast Next'],
            'values': [float(monthly_df['sales'].sum()), float(next_month_pred)]
        }
        
        # Residuals & Safety Stock
        
        # Residuals & Safety Stock
        y_pred_hist = self.model.predict(X)
        residuals = y - y_pred_hist
        suggested_stock = self._calculate_safety_stock(next_month_pred, residuals)
        
        # Financials
        avg_price = df['price'].mean() if 'price' in df.columns else None
        revenue = next_month_pred * avg_price if avg_price else None
        
        # Generate Report via GPT-5 (OpenAI)
        report = self._generate_ai_report(monthly_df, next_month_pred, self.mae, avg_price, revenue)

        result = {
            'predicted_demand': float(next_month_pred),
            'estimated_revenue': float(revenue) if revenue is not None else None,
            'suggested_stock': int(suggested_stock),
            'report': report,
            'model_info': {
                'model_used': self.model.__class__.__name__,
                'mae': self.mae,
                'rmse': self.rmse,
                'history_months': len(monthly_df)
            },
            'history': {
                'dates': monthly_df['date'].dt.strftime('%Y-%m').tolist(),
                'sales': monthly_df['sales'].tolist()
            },
            'bar_chart': bar_chart,
            'pie_chart': pie_chart
        }
        return result

    def _analyze_per_item(self, df: pd.DataFrame, item_col: str) -> Dict[str, Any]:
        """Runs analysis per item and aggregates results."""
        items = df[item_col].unique()
        
        total_demand = 0.0
        total_revenue = 0.0
        total_stock_suggested = 0
        
        item_results = []
        
        # Global history for chart (sum of all items)
        global_monthly = self._aggregate_monthly(df)
        # Bar/pie charts for item-level view will be based on top products
        
        for item in items:
            item_df = df[df[item_col] == item].copy()
            # Simple aggregation to monthly for this item
            monthly_item = self._aggregate_monthly(item_df)
            
            if monthly_item.empty: continue
            
            # Use simple heuristic if not enough data, else train model (simplified for individual items to avoid overfitting/errors on small data)
            # For robustness on item level which might be sparse, we'll favor simple heuristics or lighter models
            # But let's try to use the main logic if possible.
            
            pred = 0.0
            
            if len(monthly_item) >= self.n_lags + 2:
                 try:
                    feat = self._create_features(monthly_item)
                    X = feat.drop(columns=['sales', 'date']).values
                    y = feat['sales'].values
                    # Train a quick local model
                    if len(y) > 12: 
                        model = RandomForestRegressor(n_estimators=50, random_state=42) 
                    else: 
                        model = LinearRegression()
                    model.fit(X, y)
                    last_vals = monthly_item['sales'].iloc[-self.n_lags:].values[::-1]
                    
                    input_vec = last_vals.reshape(1, -1)
                    pred = max(0.0, float(model.predict(input_vec)[0]))
                 except Exception as e:
                    logger.warning(f"Modeling failed for item {item}: {e}. Falling back to simple average.")
                    pred = float(monthly_item['sales'].iloc[-3:].mean()) if len(monthly_item) > 0 else 0.0
            else:
                 pred = float(monthly_item['sales'].iloc[-3:].mean()) if len(monthly_item) > 0 else 0.0
            
            # Calculate item revenue
            item_price = item_df['price'].mean() if 'price' in item_df.columns else 0.0
            item_revenue = pred * item_price
            
            total_demand += pred
            total_revenue += item_revenue
            
            # Safety stock (simplified)
            stock = int(pred * 1.2) # Simple 20% buffer for item level
            total_stock_suggested += stock
            
            item_results.append({
                'item': item,
                'forecast': pred,
                'revenue': item_revenue,
                'price': item_price,
                'stock_suggested': stock,
                'total_sold': item_df['sales'].sum()
            })
            
        # Top Products
        item_results.sort(key=lambda x: x['total_sold'], reverse=True)
        top_products = item_results[:5]
        
        # Restock Recommendations (Filter items with significant forecast)
        restock_recs = [i for i in item_results if i['forecast'] > 0]
        restock_recs.sort(key=lambda x: x['forecast'], reverse=True)
        restock_recs = restock_recs[:10] # Top 10 needed
        
        # Generate aggregate report
        avg_price_global = df['price'].mean() if 'price' in df.columns else None
        # MAE calculation for the aggregate report; simplified to 0.0 for this view if not available.
        report = self._generate_ai_report(global_monthly, total_demand, 0.0, avg_price_global, total_revenue, top_products=top_products)
        
        # build charts from top_products
        bar_chart = {'labels': [], 'values': []}
        pie_chart = {'labels': [], 'values': []}
        if top_products:
            for prod in top_products:
                bar_chart['labels'].append(prod['item'])
                bar_chart['values'].append(prod['total_sold'])
                pie_chart['labels'].append(prod['item'])
                pie_chart['values'].append(prod['forecast'])

        return {
            'predicted_demand': float(total_demand),
            'estimated_revenue': float(total_revenue),
            'suggested_stock': int(total_stock_suggested),
            'report': report,
            'model_info': {
                'model_used': 'Item-Level Aggregation',
                'mae': None, # Consolidated MAE is complex to represent
                'history_months': len(global_monthly)
            },
            'history': {
                'dates': global_monthly['date'].dt.strftime('%Y-%m').tolist(),
                'sales': global_monthly['sales'].tolist()
            },
            'top_products': top_products,
            'restock_recs': restock_recs,
            'bar_chart': bar_chart,
            'pie_chart': pie_chart
        }

    def _load_and_validate_data(self, 
                                csv_path: Union[str, Path],
                                date_col: Optional[str], 
                                sales_col: Optional[str], 
                                price_col: Optional[str],
                                item_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Loads CSV, infers columns if not provided, and standardizes column names.
        Returns (DataFrame, item_col_name)
        """
        try:
            # Use utf-8-sig to handle potential BOM from Excel-saved CSVs
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            df.columns = [str(c).strip() for c in df.columns]  # Clean column names
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            try:
                df = pd.read_csv(csv_path, encoding='latin-1')
                df.columns = [str(c).strip() for c in df.columns]
            except Exception as e:
                raise ValueError(f"Could not read CSV file with utf-8 or latin-1 encoding: {e}")
        except Exception as e:
            raise ValueError(f"Could not read CSV file: {e}")

        # helper for cleaning numeric fields (strip commas/currency/etc.)
        def clean_numeric(series: pd.Series) -> pd.Series:
            # convert everything to string, strip out non-numeric except dot and minus
            cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
            return pd.to_numeric(cleaned, errors='coerce').fillna(0)

        # 1. Identify Date Column
        if date_col:
            if date_col not in df.columns:
                raise ValueError(f"Date column '{date_col}' not found.")
            used_date_col = date_col
        else:
            used_date_col = self._infer_column(df, ['date', 'timestamp', 'ds', 'time', 'day'])
            if not used_date_col:
                 used_date_col = self._scan_for_date_col(df)
        
        use_synthetic_timeline = False
        if not used_date_col:
            # For datasets without an explicit date (common in product-level exports),
            # synthesize a timeline so demand logic and charts remain functional.
            use_synthetic_timeline = True
            logger.warning("No reliable date column detected. Using synthetic daily timeline.")

        # 2. Identify Sales Column
        if sales_col:
            if sales_col not in df.columns:
                 raise ValueError(f"Sales column '{sales_col}' not found.")
            used_sales_col = sales_col
        else:
            # prefer quantity terms before generic "sales" to avoid matching strings like
            # "Sales_Rep"; also require numeric values for sales.
            used_sales_col = self._infer_column(
                df,
                ['quantity', 'qty', 'units', 'sales', 'demand', 'sold'],
                numeric_only=True
            )
            if not used_sales_col:
                # final fallback: pick first numeric column that is not the date
                nums = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in (used_date_col,)]
                if nums:
                    used_sales_col = nums[0]
                else:
                    raise ValueError("Could not detect a sales/quantity column.")
            # verify conversion yields something sensible, otherwise try other numeric
            cleaned = clean_numeric(df[used_sales_col])
            if cleaned.sum() == 0 and len(cleaned) > 0:
                # maybe we picked a wrong numeric column (all zeros); try alternatives
                for alt in df.select_dtypes(include=[np.number]).columns:
                    if alt == used_sales_col or alt == used_date_col:
                        continue
                    alt_clean = clean_numeric(df[alt])
                    if alt_clean.sum() > 0:
                        used_sales_col = alt
                        break
        

        # 3. Identify Price Column (Optional)
        used_price_col = None
        if price_col:
             if price_col in df.columns:
                 used_price_col = price_col
        else:
            used_price_col = self._infer_column(df, ['price', 'revenue', 'cost', 'amount'], exclude=[used_sales_col], numeric_only=True)

        # 4. Identify Item Column (Optional)
        used_item_col = None
        if item_col:
            if item_col in df.columns:
                used_item_col = item_col
        else:
            used_item_col = self._infer_column(
                df,
                ['item', 'product', 'sku', 'id', 'name', 'code', 'category'],
                exclude=[used_sales_col, used_date_col]
            )

        # Standardize DataFrame and clean values
        # 1. Dates parsing with fallback
        if use_synthetic_timeline:
            df['date'] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df), freq='D')
        else:
            df['date'] = pd.to_datetime(df[used_date_col], errors='coerce')
            if df['date'].isna().any():
                # attempt alternative parsing
                df['date'] = pd.to_datetime(df[used_date_col], dayfirst=True, errors='coerce')
            if df['date'].isna().any():
                # drop rows that still couldn't be parsed and log
                bad_count = df['date'].isna().sum()
                logger.warning(f"Dropped {bad_count} rows due to unparseable dates in '{used_date_col}'")
                df = df.dropna(subset=['date']).reset_index(drop=True)
                if df.empty:
                    raise ValueError(f"All rows have invalid dates in column '{used_date_col}'.")

        # 2. Numeric conversions using helper
        df['sales'] = clean_numeric(df[used_sales_col])

        if used_price_col:
            df['price'] = clean_numeric(df[used_price_col])
        else:
            df['price'] = 0.0

        # 3. Item column normalization
        if used_item_col:
            df[used_item_col] = df[used_item_col].astype(str).fillna('Unknown')
        
        # 4. Final cleanup
        df = df.sort_values('date').reset_index(drop=True)
        
        # determine currency symbol or code from data if possible
        currency = None
        # if explicit column present
        if 'currency' in df.columns:
            curr = df['currency'].dropna().astype(str)
            if not curr.empty:
                currency = curr.iloc[0].strip()
        # inspect price column strings for symbols/keywords
        if not currency and used_price_col and used_price_col in df.columns:
            sample = df[used_price_col].astype(str).head(20)
            for val in sample:
                if isinstance(val, str):
                    if '$' in val:
                        currency = '$'
                        break
                    if '₹' in val or 'Rs' in val or 'INR' in val.upper():
                        currency = '₹'
                        break
                    if '€' in val:
                        currency = '€'
                        break
                    if '£' in val:
                        currency = '£'
                        break
        if not currency:
            # default to dollar
            currency = '$'

        return df, used_item_col, currency

    def _infer_column(self, df: pd.DataFrame, keywords: List[str], exclude: List[str] = [], numeric_only: bool = False) -> Optional[str]:
        """Smart search for a column name matching one of the provided keywords.

        The routine normalizes column names by lowercasing and stripping
        non-alphanumeric characters then looks for any keyword substring.
        As a final fallback it will use difflib to find the closest match.

        If ``numeric_only`` is True the returned column must be convertible to
        numeric and contain at least one non-zero value (helps avoid choosing
        text columns like "Sales_Rep" when the keyword list includes "sales").
        """
        import difflib

        def normalize(name: str) -> str:
            return ''.join(ch for ch in name.lower() if ch.isalnum())

        def is_valid_numeric(col: str) -> bool:
            # attempt to coerce and check if any non-zero or non-na values exist
            try:
                series = pd.to_numeric(df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors='coerce')
                return series.notna().any() and (series.abs() > 0).any()
            except Exception:
                return False

        norm_map = {col: normalize(col) for col in df.columns}

        # direct substring match
        for col, norm in norm_map.items():
            if col in exclude:
                continue
            for k in keywords:
                if k.lower() in norm:
                    if numeric_only and not is_valid_numeric(col):
                        # skip this candidate if it's not numeric
                        continue
                    return col

        # as fallback try fuzzy matching on normalized names
        choices = [norm for col, norm in norm_map.items() if col not in exclude]
        if choices:
            best = difflib.get_close_matches(keywords[0], choices, n=1, cutoff=0.6)
            if best:
                # find original column with this normalized name
                for col, norm in norm_map.items():
                    if norm == best[0]:
                        if numeric_only and not is_valid_numeric(col):
                            continue
                        return col
        return None

    def _scan_for_date_col(self, df: pd.DataFrame) -> Optional[str]:
        """Aggressively tries to convert columns to datetime to find the date column."""
        for col in df.columns:
            try:
                # Ignore strongly numeric columns (e.g., IDs) to prevent epoch false positives.
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue

                # Check a larger sample with coercion and quality constraints.
                sample = df[col].dropna().astype(str).str.strip().iloc[:100]
                if sample.empty:
                    continue

                parsed = pd.to_datetime(sample, errors='coerce')
                parse_ratio = parsed.notna().mean()
                if parse_ratio < 0.7:
                    continue

                years = parsed.dropna().dt.year
                if years.empty:
                    continue

                # Keep realistic year range and temporal variability.
                realistic_ratio = ((years >= 1990) & (years <= 2100)).mean()
                if realistic_ratio < 0.7:
                    continue

                if parsed.dt.to_period('M').nunique() < 2 and len(sample) > 10:
                    continue

                return col
            except (ValueError, TypeError):
                continue
        return None

    def _aggregate_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resamples data to monthly frequency, summing sales."""
        df_sorted = df.set_index('date').sort_index()
        monthly = df_sorted['sales'].resample('ME').sum().reset_index()
        return monthly[monthly['sales'] > 0] # Filter out months with 0 sales if any

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates lag features for supervised learning."""
        df_feat = df.copy()
        for i in range(1, self.n_lags + 1):
            df_feat[f'lag_{i}'] = df_feat['sales'].shift(i)
        
        # Drop rows with NaN (the first n_lags rows)
        df_feat = df_feat.dropna().reset_index(drop=True)
        return df_feat

    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Trains a model and calculates performance metrics."""
        
        if len(y) > 20:
             self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
             self.model = LinearRegression()

        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.mae = mean_absolute_error(y, preds)
        self.rmse = np.sqrt(mean_squared_error(y, preds))

    def _predict_single(self, last_values: np.ndarray) -> float:
        """Predicts the next single value given the feature vector."""
        if self.model is None:
            return float(np.mean(last_values))
        
        input_vec = last_values.reshape(1, -1)
        pred = self.model.predict(input_vec)[0]
        return max(0.0, float(pred)) # No negative sales

    def _calculate_safety_stock(self, predicted_demand: float, residuals: np.ndarray) -> int:
        """Calculates safety stock based on prediction residuals."""
        if residuals.size == 0:
            std_dev = predicted_demand * 0.1 # Fallback 10% assumption
        else:
            std_dev = np.std(residuals)
        
        safety_stock = self.service_level_z * std_dev * np.sqrt(self.lead_time_months)
        return int(np.ceil(predicted_demand + safety_stock))

    def _simple_heuristic_analysis(self, monthly_df: pd.DataFrame, full_df: pd.DataFrame, price_col: Optional[str]) -> Dict[str, Any]:
        """Fallback analysis when data is too scarce."""
        recent_sales = monthly_df['sales'].tail(3)
        if recent_sales.empty:
             pred = 0.0
        else:
             pred = float(recent_sales.mean())
        
        avg_price = full_df['price'].mean() if 'price' in full_df.columns else None
        revenue = pred * avg_price if avg_price else None
        
        return {
            'predicted_demand': pred,
            'estimated_revenue': revenue,
            'suggested_stock': int(pred * 1.2), # Simple +20% buffer
            'report': "Insufficient data for advanced AI modeling. Using simple averages.",
            'model_info': {'model_used': 'Simple Average', 'mae': 0.0, 'rmse': 0.0, 'history_months': len(monthly_df)},
            'history': {
                'dates': monthly_df['date'].dt.strftime('%Y-%m').tolist(),
                'sales': monthly_df['sales'].tolist()
            }
        }

    def _generate_ai_report(self, df: pd.DataFrame, predicted: float, mae: float, avg_price: Optional[float], revenue: Optional[float], top_products: List[Dict] = None) -> str:
        """Generates a business intelligence report using OpenAI (GPT-5/4)."""
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return self._generate_report(df, predicted, mae, avg_price, revenue) + "\n\n(Note: AI configuration missing. Using standard functionality.)"

        try:
            client = openai.OpenAI(api_key=api_key)
            
            # Prepare context for the AI
            history_summary = df.tail(6).to_dict(orient='records') # Last 6 months
            context = {
                "historical_sales_last_6_months": history_summary,
                "forecast_next_month": predicted,
                "model_mae": mae,
                "average_unit_price": avg_price,
                "estimated_revenue_next_month": revenue,
                "top_products_summary": str(top_products[:3]) if top_products else "N/A"
            }

            prompt = f"""
            You are an expert Data Scientist and Business Consultant.
            Analyze the following demand prediction data and provide a concise, actionable executive summary.
            
            Data Context:
            {json.dumps(context, indent=2, default=str)}

            Requirements:
            1. Identify the sales trend (up/down/stable).
            2. Evaluate the forecast reliability based on MAE.
            3. Provide specific inventory or marketing recommendations based on the trend.
            4. Keep it professional, concise, and using bullet points where appropriate.
            """

            response = client.chat.completions.create(
                model="gpt-4-turbo", # Updated to stable model
                messages=[
                    {"role": "system", "content": "You are a helpful business analytics assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API Call failed: {e}")
            return self._generate_report(df, predicted, mae, avg_price, revenue) + f"\n\n(AI Analysis unavailable: {str(e)})"

    def _generate_report(self, df: pd.DataFrame, predicted: float, mae: float, avg_price: Optional[float], revenue: Optional[float]) -> str:
        """Generates a human-readable business health report (Fallback)."""
        lines = []
        
        # 1. Trend Analysis
        if len(df) >= 3:
            recent_trend = df['sales'].iloc[-1] - df['sales'].iloc[-3]
            if recent_trend > 0:
                lines.append("Trend: Sales have been trending UP over the last 3 months.")
            elif recent_trend < 0:
                lines.append("Trend: Sales have been trending DOWN over the last 3 months.")
            else:
                lines.append("Trend: Sales are relatively stable.")
        
        # 2. Prediction Context
        lines.append(f"Forecast: We predict {int(predicted)} units for next month.")
        if mae and predicted > 0:
            error_pct = (mae / predicted) * 100
            if error_pct < 10:
                cert = "High"
            elif error_pct < 25:
                cert = "Moderate"
            else:
                cert = "Low"
            lines.append(f"(Model Confidence: {cert}, MAE: {mae:.1f})")

        # 3. Financials
        if revenue:
            lines.append(f"Financials: Estimated Revenue: ${revenue:,.2f} (Avg Price: ${avg_price:.2f})")

        return "\n".join(lines)


# Helper function for backward compatibility or simple usage
def analyze_csv(csv_path: str, **kwargs) -> Dict[str, Any]:
    # Reuse predictor instances so in-memory cache survives across requests.
    # Keyed by predictor configuration for safety.
    if not hasattr(analyze_csv, "_predictor_pool"):
        analyze_csv._predictor_pool = {}

    key = (
        kwargs.get("n_lags", 3),
        kwargs.get("lead_time_months", 1.0),
        kwargs.get("service_level_z", 1.65),
    )
    predictor = analyze_csv._predictor_pool.get(key)
    if predictor is None:
        predictor = DemandPredictor(**kwargs)
        analyze_csv._predictor_pool[key] = predictor

    return predictor.analyze(csv_path)

if __name__ == '__main__':
    # Local Test
    import sys
    if len(sys.argv) > 1:
        res = analyze_csv(sys.argv[1])
        print(res)
    else:
        print("Usage: python predictor.py <path_to_csv>")
