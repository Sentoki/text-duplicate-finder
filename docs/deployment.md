# Production Deployment (Ubuntu 24.04 LTS)

This document describes how to set up automatic startup of text-duplicate-finder on Ubuntu 24.04 LTS using systemd.

## Problem

By default, uvicorn does not start automatically after a system reboot. For production environments, you need to configure the service to:

- Start automatically on system boot
- Restart on failures
- Run in the background
- Log its operation

## Solution: systemd

systemd is the standard service manager in Ubuntu. It allows you to manage applications as system services.

## Step-by-Step Instructions

### 1. Environment Preparation

Ensure the project is deployed and dependencies are installed:

```bash
cd /opt/text-duplicate-finder  # or your path
python3.12 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn sentence-transformers torch pydantic numpy
```

### 2. Create systemd Unit File

Create the service file:

```bash
sudo nano /etc/systemd/system/text-duplicate-finder.service
```

Add the following configuration:

```ini
[Unit]
Description=Text Duplicate Finder API Service
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/text-duplicate-finder
Environment="PATH=/opt/text-duplicate-finder/venv/bin"
ExecStart=/opt/text-duplicate-finder/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/var/log/text-duplicate-finder/access.log
StandardError=append:/var/log/text-duplicate-finder/error.log

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**Configuration Parameters:**

- `User=www-data` — user to run the service (change if needed)
- `WorkingDirectory` — path to the project
- `Environment="PATH=..."` — path to the virtual environment
- `ExecStart` — uvicorn startup command
- `Restart=always` — automatic restart on failure
- `RestartSec=10` — 10-second delay before restart

### 3. Create Log Directory

```bash
sudo mkdir -p /var/log/text-duplicate-finder
sudo chown www-data:www-data /var/log/text-duplicate-finder
```

### 4. Configure Permissions

If using the www-data user, grant it permissions to the project:

```bash
sudo chown -R www-data:www-data /opt/text-duplicate-finder
```

Or create a dedicated user:

```bash
sudo useradd -r -s /bin/false textdup
sudo chown -R textdup:textdup /opt/text-duplicate-finder
# And change User=textdup in the service file
```

### 5. Enable and Start Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable autostart on system boot
sudo systemctl enable text-duplicate-finder

# Start the service now
sudo systemctl start text-duplicate-finder

# Check status
sudo systemctl status text-duplicate-finder
```

The output should show `Active: active (running)`:

```
● text-duplicate-finder.service - Text Duplicate Finder API Service
     Loaded: loaded (/etc/systemd/system/text-duplicate-finder.service; enabled)
     Active: active (running) since ...
```

### 6. Verify Operation

Check that the API responds:

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Test message"}'
```

## Service Management

### Basic Commands

```bash
# Start service
sudo systemctl start text-duplicate-finder

# Stop service
sudo systemctl stop text-duplicate-finder

# Restart service
sudo systemctl restart text-duplicate-finder

# Reload configuration without restart
sudo systemctl reload text-duplicate-finder

# Check status
sudo systemctl status text-duplicate-finder

# Disable autostart
sudo systemctl disable text-duplicate-finder
```

### Viewing Logs

```bash
# Logs via journalctl (recommended)
sudo journalctl -u text-duplicate-finder -f

# Last 100 lines
sudo journalctl -u text-duplicate-finder -n 100

# Logs since a specific time
sudo journalctl -u text-duplicate-finder --since "1 hour ago"

# Direct log files
tail -f /var/log/text-duplicate-finder/access.log
tail -f /var/log/text-duplicate-finder/error.log
```

## Application Updates

When updating code:

```bash
# 1. Go to project directory
cd /opt/text-duplicate-finder

# 2. Pull updates (if using git)
sudo -u www-data git pull

# 3. Update dependencies (if changed)
sudo -u www-data venv/bin/pip install -r requirements.txt

# 4. Restart service
sudo systemctl restart text-duplicate-finder

# 5. Verify everything works
sudo systemctl status text-duplicate-finder
```

## Troubleshooting

### Service Won't Start

1. Check logs:
```bash
sudo journalctl -u text-duplicate-finder -n 50
```

2. Check permissions:
```bash
ls -la /opt/text-duplicate-finder
```

3. Verify uvicorn path is correct:
```bash
/opt/text-duplicate-finder/venv/bin/uvicorn --version
```

4. Try running manually:
```bash
cd /opt/text-duplicate-finder
source venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Port Already in Use

Check if port 8000 is free:

```bash
sudo lsof -i :8000
```

If the port is occupied, either stop the other process or change the port in the service file.

### Service Crashes After Start

Increase `RestartSec` in the service file and check logs:

```ini
RestartSec=30
```

## Monitoring

### Automatic Availability Check

You can set up cron for monitoring:

```bash
# Create script /usr/local/bin/check-textdup.sh
#!/bin/bash
if ! curl -f http://localhost:8000/embed > /dev/null 2>&1; then
    systemctl restart text-duplicate-finder
    echo "$(date): Service restarted" >> /var/log/text-duplicate-finder/monitor.log
fi

# Make executable
sudo chmod +x /usr/local/bin/check-textdup.sh

# Add to crontab (every 5 minutes)
sudo crontab -e
*/5 * * * * /usr/local/bin/check-textdup.sh
```

### Resource Monitoring

```bash
# Memory and CPU usage
systemctl status text-duplicate-finder

# Detailed information
ps aux | grep uvicorn

# Via htop (install if needed)
sudo apt install htop
htop  # find uvicorn process
```

## Security

### Firewall (UFW)

If the service should only be accessible within the network:

```bash
# Allow access only from local network
sudo ufw allow from 192.168.1.0/24 to any port 8000

# Or only from specific IP
sudo ufw allow from 192.168.1.100 to any port 8000
```

### Reverse Proxy (nginx)

For production, using nginx is recommended:

```bash
sudo apt install nginx

# /etc/nginx/sites-available/text-duplicate-finder
server {
    listen 80;
    server_name your-server.local;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

sudo ln -s /etc/nginx/sites-available/text-duplicate-finder /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Performance

### Multiple Worker Processes

For high-load systems, you can run multiple workers:

Change `ExecStart` in the service file:

```ini
ExecStart=/opt/text-duplicate-finder/venv/bin/uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

Number of workers = number of CPU cores.

**Warning:** Each worker loads its own copy of the model (~1.5 GB), account for RAM!

## Summary

After configuration, the service will:
- [OK] Automatically start on Ubuntu boot
- [OK] Restart on failures
- [OK] Log its operation
- [OK] Be managed with standard systemd commands

On server reboot, the service will start automatically within ~30-60 seconds after system boot.
