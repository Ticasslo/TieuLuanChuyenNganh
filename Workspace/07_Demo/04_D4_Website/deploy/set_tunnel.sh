#!/usr/bin/env bash
# Bật Cloudflare Tunnel cho website: hỏi token (không hiện ra màn hình, không lưu vào lịch sử lệnh), ghi tệp .env
# quyền 600 cạnh compose.yaml, đóng cổng 80/443 ra Internet (Caddy chỉ nghe 127.0.0.1, cloudflared nối vào qua
# mạng Docker), rồi chờ tunnel đăng ký kết nối với Cloudflare.
# Chạy trên VPS: bash set_tunnel.sh        Quay lại chạy thẳng bằng IP (mở cổng 80): bash set_tunnel.sh --off
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--off" ]]; then
  rm -f .env
  sudo docker compose --profile tunnel down --remove-orphans
  sudo docker compose up -d
  echo "Đã tắt tunnel, website chạy lại ở cổng 80."
  exit 0
fi

read -rsp "Dán token tunnel (chuỗi bắt đầu bằng eyJ, không hiện khi dán) rồi Enter: " token
echo
token="${token//[[:space:]]/}"
if [[ "$token" != eyJ* ]]; then
  echo "Token phải bắt đầu bằng eyJ (chỉ lấy phần sau --token trong lệnh Cloudflare đưa)." >&2
  exit 1
fi

umask 077
cat > .env <<EOF
COMPOSE_PROFILES=tunnel
TUNNEL_TOKEN=$token
DEMO_BIND=127.0.0.1
DEMO_CLIENT_IP_HEADER=CF-Connecting-IP
EOF
chmod 600 .env
unset token

sudo docker compose up -d --remove-orphans
for _ in $(seq 1 30); do
  if sudo docker compose logs tunnel 2>&1 | grep -q "Registered tunnel connection"; then
    echo "Tunnel đã kết nối Cloudflare. Cổng 80/443 không còn mở ra Internet."
    sudo docker compose ps --format "{{.Name}} {{.Status}}"
    exit 0
  fi
  sleep 2
done
echo "Tunnel chưa kết nối sau 60 giây. 20 dòng log cuối:" >&2
sudo docker compose logs --tail 20 tunnel >&2
exit 1
