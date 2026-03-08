from django import forms
from django.core.exceptions import ValidationError


class UploadCSVForm(forms.Form):
    file = forms.FileField(label='CSV file', help_text='Upload a CSV file (max 10MB).')

    def clean_file(self):
        f = self.cleaned_data.get('file')
        if not f:
            return f
        if not f.name.lower().endswith('.csv'):
            raise forms.ValidationError('Only CSV files are allowed.')
        # Limit file size to 10MB
        if f.size > 10 * 1024 * 1024:
            raise forms.ValidationError('File size must be under 10MB.')
        return f
