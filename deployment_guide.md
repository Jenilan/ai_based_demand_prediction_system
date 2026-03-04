# Deployment Guide: AI Demand Prediction System

This guide outlines the steps to deploy your Django application to a production server (Linux-based) using Gunicorn and Nginx.

## Prerequisites
- A Linux server (Ubuntu 22.04 recommended)
- Python 3.10+ installed
- PostgreSQL (recommended for production) or SQLite
- OpenAI API Key

## 1. Environment Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd AI_DEMEND_PREDICTION_SYSTEM
    ```

2.  **Create Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file:
    ```bash
    # .env
    DEBUG=False
    SECRET_KEY=your-secure-secret-key
    ALLOWED_HOSTS=yourdomain.com,203.0.113.1
    OPENAI_API_KEY=sk-your-openai-key
    DATABASE_URL=postgres://user:password@localhost:5432/dbname
    ```

5.  **Collect Static Files**
    ```bash
    python manage.py collectstatic --noinput
    ```

6.  **Run Migrations**
    ```bash
    python manage.py migrate
    ```

## 2. Gunicorn Setup

Gunicorn is the WSGI HTTP Server that will run your Django code.

1.  **Test Gunicorn**
    ```bash
    gunicorn --bind 0.0.0.0:8000 demand_pro.wsgi:application
    ```
    Visit `http://<server-ip>:8000` to verify it loads.

2.  **Create Systemd Service** (to keep it running)
    Create file `/etc/systemd/system/gunicorn.service`:

    ```ini
    [Unit]
    Description=gunicorn daemon
    After=network.target

    [Service]
    User=ubuntu
    Group=www-data
    WorkingDirectory=/home/ubuntu/AI_DEMEND_PREDICTION_SYSTEM
    ExecStart=/home/ubuntu/AI_DEMEND_PREDICTION_SYSTEM/venv/bin/gunicorn \
          --access-logfile - \
          --workers 3 \
          --bind unix:/home/ubuntu/AI_DEMEND_PREDICTION_SYSTEM/demand_pro.sock \
          demand_pro.wsgi:application

    [Install]
    WantedBy=multi-user.target
    ```

3.  **Start and Enable Gunicorn**
    ```bash
    sudo systemctl start gunicorn
    sudo systemctl enable gunicorn
    ```

## 3. Nginx Setup

Nginx acts as the reverse proxy and handles static files.

1.  **Install Nginx**
    ```bash
    sudo apt update
    sudo apt install nginx
    ```

2.  **Configure Nginx**
    Create file `/etc/nginx/sites-available/demand_pro`:

    ```nginx
    server {
        listen 80;
        server_name yourdomain.com 203.0.113.1;

        location = /favicon.ico { access_log off; log_not_found off; }
        
        # Serve Static Files
        location /static/ {
            alias /home/ubuntu/AI_DEMEND_PREDICTION_SYSTEM/staticfiles/;
        }

        # Serve Media Files (Uploads)
        location /media/ {
            alias /home/ubuntu/AI_DEMEND_PREDICTION_SYSTEM/media/;
        }

        # Proxy to Gunicorn
        location / {
            include proxy_params;
            proxy_pass http://unix:/home/ubuntu/AI_DEMEND_PREDICTION_SYSTEM/demand_pro.sock;
        }
    }
    ```

3.  **Enable Site**
    ```bash
    sudo ln -s /etc/nginx/sites-available/demand_pro /etc/nginx/sites-enabled
    sudo nginx -t
    sudo systemctl restart nginx
    ```

## 4. HTTPS (SSL)

Use Certbot to secure your site:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

## Troubleshooting
- **Logs**: Check `sudo journalctl -u gunicorn` for app errors.
- **Nginx**: Check `/var/log/nginx/error.log`.
- **OpenAI Errors**: Ensure `OPENAI_API_KEY` is in the `.env` file and visible to Gunicorn (you may need to add `EnvironmentFile` directive in the systemd service).
