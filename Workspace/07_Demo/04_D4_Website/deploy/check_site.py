# Kiểm một bản website đang chạy (laptop hoặc VPS): /api/health đủ cờ, mọi mô hình "nếu... thì..." chạy được
# ở vài lưu vực và ngày (máy chủ tự so dự báo gốc với tệp phát lại, lệch thì trả 422), đo thời gian từng lần,
# và tùy chọn so JSON các API chỉ đọc với một bản khác (VD laptop) bằng sha256.
# Chạy: python check_site.py --url https://<tên miền> [--compare http://127.0.0.1:8060]
# Mỗi lần "nếu... thì..." cách nhau đủ để không chạm giới hạn lần/phút của máy chủ (--per-minute).

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request

from urllib.parse import urlencode

READ_ONLY = ("/api/catalog", "/api/site", "/api/basins", "/api/highlights", "/api/leaderboard", "/api/by-lead",
             "/api/c2-seeds", "/api/d-pairs", "/api/env", "/api/basins-own", "/api/rivers")


def read_only_paths(catalog):
    # Các API chỉ đọc giao diện gọi, kèm tham số của mô hình mặc định, hạn 1, lưu vực và ngày đầu kỳ
    run, basin, date = catalog["default_run"], str(catalog["ids"][0]), catalog["dates"][0]
    with_args = {"/api/flood-summary": dict(run=run, lead=1), "/api/confusion": dict(run=run, lead=1),
                 "/api/basin-scores": dict(run=run, lead=1), "/api/day": dict(date=date, run=run, lead=1),
                 "/api/days": dict(run=run, lead=1, start=date, n=60), "/api/flood": dict(run=run, basin=basin, lead=1),
                 "/api/chart": dict(basin=basin, lead=1, run=run), "/api/fan": dict(basin=basin, run=run, date=date),
                 "/api/env-days": dict(layer="wind", start=date, n=60)}
    return list(READ_ONLY) + [f"{path}?{urlencode(args)}" for path, args in with_args.items()]


# Cloudflare chặn User-Agent mặc định "Python-urllib" (Browser Integrity Check) => gửi tên riêng của bộ kiểm
HEADERS = {"User-Agent": "demo-site-check/1.0"}


def get(base, path):
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=HEADERS), timeout=60) as response:
        return response.read()


def post(base, path, body):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={**HEADERS, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def main():
    parser = argparse.ArgumentParser(description="Kiểm website demo đang chạy.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--compare")
    parser.add_argument("--per-minute", type=int, default=12)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    health = json.loads(get(base, "/api/health"))
    if not health.get("ok"):
        raise SystemExit(f"/api/health không ok: {health}")
    print("health:", health)

    catalog = json.loads(get(base, "/api/catalog"))
    live = [m for m in catalog["models"] if m.get("what_if")]
    if not live:
        raise SystemExit("Không có mô hình 'nếu... thì...' nào được bật.")
    ids, dates = catalog["ids"], catalog["dates"]
    # Ba lưu vực rải đều và ba ngày phát hành rải trong kỳ test (H giữ trong khoảng ngày phát hành của mô hình)
    basins = [str(ids[i]) for i in (0, len(ids) // 2, len(ids) - 1)]
    gap = 60 / args.per_minute + 0.5
    failures = []
    for model in live:
        start = model.get("origin_start", dates[0])
        end = model.get("origin_end", dates[-1])
        window = [d for d in dates if start <= d <= end]
        for basin, date in zip(basins, (window[10], window[len(window) // 2], window[-10])):
            for rain, temp in ((1, 0), (1.5, 2)):
                tic = time.perf_counter()
                status, reply = post(base, "/api/what-if", dict(run=model["run"], basin=basin, date=date,
                                                                rain_multiplier=rain, temperature_delta=temp))
                took = time.perf_counter() - tic
                ok = status == 200
                print(f"{model['run']:<40} basin {basin:>4} {date} x{rain} +{temp}: {status} {took:5.2f} s")
                if not ok:
                    failures.append((model["run"], basin, date, rain, temp, status, reply.get("error")))
                time.sleep(gap)

    if args.compare:
        other = args.compare.rstrip("/")
        for path in read_only_paths(catalog):
            same = hashlib.sha256(get(base, path)).hexdigest() == hashlib.sha256(get(other, path)).hexdigest()
            print(f"{path:<60} {'trùng' if same else 'KHÁC'}")
            if not same:
                failures.append((path, "khác bản so sánh"))

    if failures:
        raise SystemExit("Lỗi:\n" + "\n".join(map(str, failures)))
    print(f"Đạt: {len(live)} mô hình 'nếu... thì...', mọi lần chạy trả 200.")


if __name__ == "__main__":
    main()
