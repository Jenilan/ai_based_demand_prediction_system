# AI Demand Prediction System

A Django-based application for predicting product demand using historical data and Machine Learning.

## Features
- Upload CSV sales data
- Predict future demand using Random Forest / Linear Regression
- Generate revenue estimates and restock suggestions
- Item-level and Aggregate analysis
- AI-powered business insights (requires OpenAI API key)
- Production-ready with Docker support

## Quick Start (Local Development)

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd AI_DEMEND_PREDICTION_SYSTEM
   ```

2. **Set up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Migrations**
   ```bash
   python manage.py migrate
   ```

5. **Start Server**
   ```bash
   python manage.py runserver
   ```
   Visit `http://localhost:8000`

## Production Deployment

### Environment Variables
Copy `.env.example` to `.env` and configure:

| Variable | Description | Required |
|----------|-------------|----------|
| `DJANGO_SECRET_KEY` | **Required**. A long random string. | Yes |
| `DJANGO_DEBUG` | **Required**. Set to `False` for production. | Yes |
| `DJANGO_ALLOWED_HOSTS` | **Required**. Comma-separated list of domains. | Yes |
| `DATABASE_URL` | **Required**. Connection string for your DB. | Yes |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | Trusted origins for CSRF (if behind proxy) | No |
| `DJANGO_ENABLE_SSL` | Set to `True` to force HTTPS. | No |
| `OPENAI_API_KEY` | OpenAI API key for AI reports | No |

### Docker Deployment
```bash
docker-compose up --build
```

### Manual Deployment
See `deployment_guide.md` for detailed Linux server setup.

## API Endpoints
- `GET /health/` - Health check
- `GET /` - Upload form
- `POST /` - Process upload
- `GET /results/?file=...` - View results

## Testing
```bash
python manage.py test
```

## Security Notes
- File uploads limited to 10MB CSV files
- Path traversal protection implemented
- Environment variables for sensitive data
- SSL/TLS support for production

### 2. Docker Deployment
The project includes a `Dockerfile` and `docker-compose.yml` for containerized deployment.

**Build and Run:**
```bash
docker-compose up --build
```
*Note: Ensure you have a `.env` file in the root directory with the variables listed above.*

### 3. Collecting Static Files
If running without Docker or needing to manually collect static files:
```bash
python manage.py collectstatic
```
