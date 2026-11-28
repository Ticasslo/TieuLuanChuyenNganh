# Đưa website demo lên VPS (D4)

Hướng dẫn dựng lại và vận hành website demo trên máy ảo Oracle Cloud Always Free (ARM64), công khai tại `https://ticasslo.dev` qua Cloudflare Tunnel. Mã chạy là đúng mã của bản laptop (`04_D4_Website/server.py` dùng lại `create_app` của D3, bộ dự báo CPU `DemoRuntime/`); mọi tệp triển khai nằm trong thư mục này.

Repo công khai => IP máy, token tunnel, ID tài khoản Cloudflare không ghi vào repo. Trong tài liệu, `<IP máy>` là IP công khai xem ở Oracle Console (Compute → Instances).

## 1. Kiến trúc

Người xem => Cloudflare (HTTPS, lọc DDoS, giấu IP gốc) => `cloudflared` trên VPS (tự mở kết nối ra Cloudflare, không cần cổng vào) => Caddy (nén, giới hạn kích thước yêu cầu, header bảo mật) => gunicorn chạy `server:from_env()`.

| Thành phần | Tệp | Ghi chú |
|---|---|---|
| Ảnh app | `Dockerfile` | Python 3.12.10, hai venv: `/opt/venv-c` (ứng dụng + 5 lõi C2, NeuralHydrology `freeze-C` `db6dc1f`), `/opt/venv-h` (tiến trình suy luận H, `freeze-H` `6f68f47`); torch 2.11.0+cpu; phiên bản thư viện khóa theo `requirements-lock.txt` (lấy từ môi trường laptop) |
| Ba dịch vụ | `compose.yaml` | `app` (không mở cổng ra ngoài, ổ gốc chỉ đọc, người dùng không phải root), `caddy`, `tunnel` (chỉ bật khi `COMPOSE_PROFILES=tunnel`) |
| Proxy | `Caddyfile` | Nén zstd/gzip trừ PMTiles (đọc theo Range), body tối đa 16 KB, CSP, bỏ header `Server` |
| Bật tunnel | `set_tunnel.sh` | Hỏi token không hiện ra màn hình, ghi `.env` quyền 600, đóng cổng 80/443, chờ tunnel kết nối |
| Đóng gói dữ liệu | `pack_vps.py` | Gói website + nền bản đồ + địa hình + dữ liệu thô LamaH chỉ 451 lưu vực, kèm `SHA256SUMS` |
| Kiểm | `check_site.py` | `/api/health`, mọi mô hình "nếu... thì..." ở 3 lưu vực × 2 kịch bản, so 16 API chỉ đọc với một bản khác |

Trên VPS: mã ở `/srv/demo/code` (giữ cấu trúc `03_D3_LamaHCEExt_Demo/`, `04_D4_Website/`, `DemoRuntime/` như `Workspace/07_Demo`), dữ liệu ở `/srv/demo/data` (gắn chỉ đọc vào app tại `/data`).

## 2. Cần có

- Máy `VM.Standard.A1.Flex` 2 OCPU / 12 GB, Ubuntu 22.04 hoặc 24.04 aarch64 (wheel torch aarch64 cần glibc ≥ 2.28, trang download.pytorch.org/whl/cpu). Hạn mức Always Free và tiêu chí thu hồi máy rảnh: docs.oracle.com *Always Free Resources*.
- Khóa SSH riêng của máy (`~/.ssh/ssh-key-2026-09-07.key`); người dùng `ubuntu`.
- Tên miền đặt nameserver ở Cloudflare (gói Free). `ticasslo.dev` đăng ký ở name.com qua GitHub Student Developer Pack, hết hạn 27/11/2027.
- Trên laptop: gói website (`D:\TLCN\Website_Bundle`), nền bản đồ `downloads/cee_z11.pmtiles`, địa hình `downloads/terrain_z11.pmtiles`, dữ liệu thô `downloads/lamah_raw/LamaH_extended`, môi trường `.venv` của website, công cụ `downloads/pmtiles_tool/pmtiles.exe` (go-pmtiles 1.31.2).

## 3. Dựng lại từ đầu

Thứ tự dưới đây là thứ tự đã chạy thành công ngày 28/11/2026. Bước sau chỉ làm khi bước trước đạt.

**Bước 0 - Dữ liệu lớp bản đồ và địa hình (laptop, chỉ làm khi chưa có).**
- Mạng sông HydroRIVERS v1.0 (châu Âu, shapefile): tải `https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_eu_shp.zip` vào `downloads/hydrorivers/` rồi giải nén. Giấy phép HydroSHEDS v1 cho dùng tự do, kèm câu ghi nguồn bắt buộc (đã đặt ở trang "Về dự án").
- Lớp khí tượng, đất, thực vật, đặc điểm lưu vực, phần riêng của lưu vực, mạng sông, đường tuyết: `.venv/Scripts/python.exe -I -X utf8 build_map_layers.py --raw downloads/lamah_raw/LamaH_extended --bundle D:/TLCN/Website_Bundle --rivers downloads/hydrorivers/shp/HydroRIVERS_v10_eu_shp/HydroRIVERS_v10_eu.shp` => `env.json` (kèm `snowline`) + `env/` (10 tệp, 26,4 MB), `basins_own.geojson` (451 mảnh, hở và chồng 0,000%), `rivers.geojson` (7.753 đoạn, 1,8 MB; chiều vẽ xuôi dòng 29.201/29.201) trong gói website; script tự đọc lại 40 ô ngẫu nhiên bằng mô-đun `csv` và dừng nếu lệch. Lần chạy khoảng 8 phút. Phải dừng máy chủ chạy thử trên laptop trước (Windows khóa tệp `.npy` đang mở).
- Địa hình Mapterhorn cắt theo khung nền bản đồ, tới mức phóng 11: `downloads/pmtiles_tool/pmtiles.exe extract https://download.mapterhorn.com/planet.pmtiles downloads/terrain_z11.pmtiles --bbox=7.5,45.8,17.8,50.9 --maxzoom=11` (347 MB; mức 12 là 1,3 GB, đo bằng `--dry-run`). Ghi công `© Mapterhorn` (metadata của tệp planet) hiện ở góc bản đồ; nguồn và giấy phép từng nước: mapterhorn.com/attribution.

**Bước 1 - Đóng gói dữ liệu (laptop).** Trong `04_D4_Website`: `.venv/Scripts/python.exe -I -X utf8 deploy/pack_vps.py --bundle D:/TLCN/Website_Bundle --raw downloads/lamah_raw/LamaH_extended --pmtiles downloads/cee_z11.pmtiles --terrain downloads/terrain_z11.pmtiles --out D:/TLCN/VPS_Pack`. Kết quả lần chạy: 1.095 tệp, 2,87 GiB. Cần `-X utf8` vì `-I` bỏ qua `PYTHONIOENCODING`, Windows không in được chữ tiếng Việt.

**Bước 2 - Chuẩn bị máy.**
- Đăng nhập: `ssh -i ~/.ssh/ssh-key-2026-09-07.key ubuntu@<IP máy>`.
- Cài Docker Engine + Compose plugin theo kho chính thức (docs.docker.com *Install Docker Engine on Ubuntu*); lần cài: Docker 29.10.1, Compose v5.8.1. Thêm `ubuntu` vào nhóm `docker`.
- Kiểm SSH chỉ cho khóa: `sudo sshd -T | grep passwordauthentication` phải ra `no`.
- Tạo thư mục: `sudo mkdir -p /srv/demo/data /srv/demo/code && sudo chown -R ubuntu:ubuntu /srv/demo`.
- Không cần mở cổng 80/443 ở Oracle Security List hay iptables khi dùng tunnel (chỉ cần cổng 22).

**Bước 3 - Chép dữ liệu và kiểm.** Từ `D:/TLCN/VPS_Pack` trên laptop (Git Bash): `tar -cf - . | gzip -1 | ssh -i ~/.ssh/ssh-key-2026-09-07.key ubuntu@<IP máy> 'tar -xzf - -C /srv/demo/data'` (lần chạy khoảng 10 phút). Trên VPS: `cd /srv/demo/data && sha256sum --quiet -c SHA256SUMS` không in gì là đạt.

**Bước 4 - Chép mã.** Từ `Workspace/07_Demo` trên laptop: `tar --exclude=__pycache__ -cf - 03_D3_LamaHCEExt_Demo/app.py DemoRuntime/forecast.py DemoRuntime/h_cpu.py 04_D4_Website/server.py 04_D4_Website/static 04_D4_Website/deploy | ssh -i ~/.ssh/ssh-key-2026-09-07.key ubuntu@<IP máy> 'tar -xf - -C /srv/demo/code'`. Nếu sửa `set_tunnel.sh` trên Windows, trên VPS chạy `sed -i "s/\r$//" set_tunnel.sh` để bỏ CRLF.

**Bước 5 - Dựng ảnh và chạy thử bằng IP.** Trên VPS: `cd /srv/demo/code/04_D4_Website/deploy && sudo docker compose up -d --build`. Dựng lần đầu khoảng 9 phút (tải torch, cài hai bản NeuralHydrology). Khi chưa có `.env`, Caddy mở cổng 80 (HTTP), dùng để kiểm trước khi bật tunnel (cần mở tạm cổng 80 ở Security List và iptables nếu muốn gọi từ ngoài).

**Bước 6 - Kiểm kết quả trên ARM (laptop).** Bật máy chủ website trên laptop (cổng 8060, `server.py --bundle D:/TLCN/Website_Bundle --pmtiles downloads/cee_z11.pmtiles --terrain downloads/terrain_z11.pmtiles --raw downloads/lamah_raw/LamaH_extended --h-python downloads/.venv-h/Scripts/python.exe`) rồi chạy `.venv/Scripts/python.exe -I -X utf8 deploy/check_site.py --url <địa chỉ VPS> --compare http://127.0.0.1:8060`. Đạt khi in "Đạt: 7 mô hình" và cả 18 API "trùng". Máy chủ D3 tự so dự báo gốc của mỗi lần "nếu... thì..." với tệp phát lại (`03_D3_LamaHCEExt_Demo/app.py`, hàm `what_if`), lệch quá ngưỡng thì trả 422 => 42 lần 200 nghĩa là bộ dự báo trên ARM khớp tệp phát lại.

**Bước 7 - Tạo tunnel (Cloudflare Dashboard).**
1. Cloudflare Dashboard → Networking → Tunnels → Create a tunnel → Cloudflared, đặt tên (đang dùng `vps-demo`).
2. Ở bước cài đặt chọn Docker, chỉ chép phần token sau `--token` (bắt đầu `eyJ`); không chạy lệnh `docker run` Cloudflare đưa. Token là bí mật: không dán vào chat, không ghi vào repo.
3. Trang này chỉ cho Continue khi tunnel đã kết nối => làm bước 8 trước rồi quay lại.

**Bước 8 - Bật tunnel trên VPS.** Từ PowerShell của mình (cần terminal thật để nhập token ẩn): `ssh -t -i $HOME\.ssh\ssh-key-2026-09-07.key ubuntu@<IP máy> "bash /srv/demo/code/04_D4_Website/deploy/set_tunnel.sh"`, dán token, Enter. Đạt khi in "Tunnel đã kết nối Cloudflare". Script ghi `.env` gồm `COMPOSE_PROFILES=tunnel`, `TUNNEL_TOKEN`, `DEMO_BIND=127.0.0.1` (Caddy chỉ nghe nội bộ), `DEMO_CLIENT_IP_HEADER=CF-Connecting-IP`.

**Bước 9 - Thêm route (Cloudflare Dashboard).** Tunnel → Edit → Published application routes → Add: Subdomain để trống, Domain `ticasslo.dev`, Path để trống, Service URL `http://caddy:80`, phần Additional application settings để mặc định. Cloudflare tự tạo CNAME `ticasslo.dev` => `<ID tunnel>.cfargotunnel.com`.

**Bước 10 - Kiểm công khai.** `check_site.py --url https://ticasslo.dev --compare http://127.0.0.1:8060` đạt; gọi thẳng `http://<IP máy>` và `https://<IP máy>` phải bị từ chối; `sudo reboot` rồi kiểm trang tự lên lại.

## 4. Việc thường làm

| Việc | Cách làm |
|---|---|
| Sửa giao diện hoặc `server.py` | Chép lại mã (bước 4) rồi `sudo docker compose up -d --build` trong `deploy/` (chỉ dựng lại lớp mã, vài phút) |
| Đổi dữ liệu (gói website mới) | Đóng gói lại (bước 1), chép (bước 3; chỉ phần mới cũng được, VD `tar -cf - bundle/env bundle/env.json terrain.pmtiles SHA256SUMS`), kiểm `SHA256SUMS`, rồi `sudo docker compose restart app` |
| Xem log | `sudo docker compose logs --tail 100 app` (tương tự `caddy`, `tunnel`); log xoay vòng 5 × 10 MB mỗi dịch vụ |
| Xem trạng thái | `sudo docker compose ps`; `app` phải `healthy` (kiểm `/api/health` mỗi 30 giây) |
| Khởi động lại | `sudo docker compose restart`; sau `sudo reboot` cả ba dịch vụ tự lên (`restart: unless-stopped`) |
| Tắt tunnel, chạy lại bằng IP | `bash set_tunnel.sh --off` (xóa `.env`, mở cổng 80) |
| Đổi token tunnel | Chạy lại `set_tunnel.sh` với token mới |
| Đổi tên miền | Thêm route mới trong tunnel (bước 9); phía VPS không đổi gì |

## 5. Lỗi đã gặp và cách tránh

| Hiện tượng | Nguyên nhân | Cách xử lý trong mã |
|---|---|---|
| Tiến trình H không thấy thư viện trên Linux | `DemoRuntime/h_cpu.py` gọi `Path(python).resolve()`; venv thường trên Linux là liên kết mềm => trỏ về Python gốc | `Dockerfile` tạo venv bằng `--copies` |
| `sha256sum -c` không đọc được tên tệp | `write_text` trên Windows ghi CRLF | `pack_vps.py` ghi `newline="\n"` |
| Caddy khởi động lại liên tục: "server block without any key" | Compose đặt `DEMO_DOMAIN` thành chuỗi rỗng, giá trị mặc định trong Caddyfile không dùng | Mặc định `:80` đặt ở `compose.yaml` |
| Cổng vẫn mở dù đã chặn ở iptables | Cổng do Docker mở đi qua DNAT, không qua chuỗi INPUT | Đóng bằng `DEMO_BIND=127.0.0.1`, không dựa vào iptables |
| Giới hạn "nếu... thì..." gộp mọi người làm một | Sau tunnel, Caddy chỉ thấy IP của `cloudflared` | App đọc `CF-Connecting-IP` khi đặt `DEMO_CLIENT_IP_HEADER` (thử: IP A lần 13 => 429, IP B vẫn 200) |
| Script Python bị 403 qua tên miền | Cloudflare chặn User-Agent `Python-urllib` | `check_site.py` gửi User-Agent riêng; trình duyệt không bị ảnh hưởng |
| `apt-get` báo khóa dpkg | `unattended-upgrades` đang chạy | `apt-get -o DPkg::Lock::Timeout=600` |

## 6. Kết quả kiểm (28/11/2026)

- Qua `https://ticasslo.dev`: 42/42 lần "nếu... thì..." (7 mô hình × 3 lưu vực × 2 kịch bản) trả 200, 0,6-1 giây mỗi lần (lần đầu nạp mô hình 2-5 giây); 16 API chỉ đọc trùng từng byte bản laptop; nền bản đồ trả 206 theo Range.
- RAM app khoảng 1 GB sau khi nạp đủ mô hình (máy 12 GB).
- Trình duyệt ở 1920 px và 390 px: hiển thị đúng, không cuộn ngang; đổi bài toán, phát, "nếu... thì...", đổi ngôn ngữ dưới CSP: 0 lỗi console, 0 lỗi HTTP.
- Gọi thẳng IP ở cổng 80/443 bị từ chối; iptables chỉ mở cổng 22; SSH chỉ nhận khóa (7 ngày trước khi dựng có 5.953 lượt dò mật khẩu, đều bị từ chối).
- `sudo reboot` => app, Caddy, tunnel tự lên, trang trả 200 trong vòng 1 phút.

- Lớp bản đồ (thêm 28/11/2026): qua `https://ticasslo.dev` 18/18 API chỉ đọc trùng bản laptop (thêm `/api/env`, `/api/env-days`), địa hình trả 206 theo Range; trình duyệt dưới CSP: lớp độ cao ở chế độ 3D, bóng núi tải đủ, 0 yêu cầu lỗi; laptop: phát ×10 qua 542 ngày với lớp mưa, 0 lỗi console; 390 px không cuộn ngang.
- Cloudflare tự chèn script thống kê (`static.cloudflareinsights.com/beacon.min.js`); CSP `script-src 'self'` chặn nên trang không tải script ngoài, chỉ có một dòng lỗi console. Tắt ở Cloudflare Dashboard => tên miền => Web Analytics (hoặc Speed => Settings => RUM) để hết dòng lỗi.
- Bộ nhớ đệm: máy chủ gửi `no-cache` cho mã trang (`/static/js`, `/static/css`, tên tệp không đổi giữa các lần triển khai) và `max-age=604800, public` cho `/static/vendor`, `/plotly.js`; Cloudflare vẫn hỏi lại máy chủ (`cf-cache-status: REVALIDATED`) nhưng ghi đè header gửi trình duyệt thành `max-age=14400` theo thiết lập Browser Cache TTL mặc định 4 giờ => trình duyệt có thể giữ JS cũ tới 4 giờ sau khi triển khai. Đặt Cloudflare Dashboard => tên miền => Caching => Configuration => Browser Cache TTL = "Respect Existing Headers".
- Analytics của Cloudflare tính cả mã 503 `busy` của `/api/what-if` (giới hạn số lần chạy đồng thời, giao diện báo bận và đếm ngược 5 giây trên nút; `check_site.py` gọi dồn dập nên gặp) và 502 trong khoảng vài giây khi dựng lại ảnh app; log Caddy không ghi truy cập, log app chỉ giữ từ lần khởi động gần nhất.

## 7. Rủi ro còn lại

- Oracle có thể thu hồi máy Always Free rảnh (7 ngày CPU, mạng, bộ nhớ đều dưới 20%, docs.oracle.com *Always Free Resources*) => giữ hướng dẫn này, dữ liệu đóng gói trên laptop, và bản laptop làm dự phòng ngày bảo vệ.
- Tên miền miễn phí tới 27/11/2027; sau đó gia hạn ở name.com ($22,99/năm theo trang quản lý tên miền) hoặc đổi tên miền (bước 9).
