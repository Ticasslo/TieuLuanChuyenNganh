// Nội dung "Về dự án" (chuỗi tĩnh của trang, không chứa dữ liệu người dùng nhập).
const TEXT = {
  vi: `
<div class="cols">
  <div>
    <h3>Bài toán</h3>
    <p>Lưu lượng (discharge) là lượng nước chảy qua mặt cắt sông mỗi giây, đơn vị m³/s. Trang này dự báo <b>lưu lượng lớn nhất trong ngày</b> (qmax) tại từng trạm đo, rồi so với số đo thật. Đầu ra là lưu lượng, không phải mực nước hay vùng ngập.</p>
    <p>Đề tài là nghiên cứu so sánh: cùng dữ liệu, cùng số tham số (khoảng 84.800, lệch dưới 1%), cùng cách huấn luyện 30 epoch, đổi kiến trúc LSTM, GRU, Transformer, S4D và Mamba (State Space Model) để xem kiến trúc nào dự báo tốt hơn.</p>
  </div>
  <div>
    <h3>Dữ liệu</h3>
    <dl>
      <dt>Lưu vực</dt><dd>{basins} lưu vực của bộ Extended LamaH-CE theo ngày (Zenodo 17119635, CC BY-NC 4.0)</dd>
      <dt>Huấn luyện</dt><dd>2003–2009</dd>
      <dt>Chọn mô hình</dt><dd>2010–2013 (validation)</dd>
      <dt>Kiểm tra</dt><dd>2014–2017, số liệu hiện trên trang</dd>
      <dt>Đầu vào</dt><dd>Khí tượng ERA5-Land, E-OBS, MSWEP, GLEAM; dự báo ECMWF HRES; dự báo GEFSv12 (bài toán 7 ngày); lưu lượng các ngày trước; đặc điểm lưu vực</dd>
    </dl>
  </div>
  <div>
    <h3>Cách đọc</h3>
    <ul>
      <li><b>Bốn mức lưu lượng</b> so với chính trạm đó trong 1981–2013: thấp, trung bình, cao, rất cao (2% ngày cao nhất).</li>
      <li><b>Ngưỡng lũ chu kỳ lặp lại</b> 1,5 / 2 / 5 / 10 năm, ước lượng bằng phân phối Gumbel (L-moments) trên đỉnh lũ năm 1981–2013.</li>
      <li><b>NSE</b> (Nash–Sutcliffe): 1 là khớp hoàn toàn, 0 là bằng việc đoán trung bình. <b>KGE</b> (Kling–Gupta) gộp tương quan, độ lệch trung bình và độ biến động.</li>
      <li>Mốc so sánh: <b>BiasCast</b> (Konold và cs., HESS 2026, bài cơ sở của đề tài) và <b>giữ nguyên hôm nay</b> (persistence: ngày mai bằng hôm nay).</li>
    </ul>
  </div>
</div>
<div class="cols" style="margin-top:22px">
  <div>
    <h3>Ba bài toán</h3>
    <ul>
      <li><b>Ngày mai (C2):</b> 8 cấu hình của 5 kiến trúc (cùng cỡ hoặc lớn gấp 3), mỗi cấu hình 3 lần huấn luyện; trang hiện lần có NSE validation cao nhất.</li>
      <li><b>7 ngày tới (H):</b> LSTM và Mamba với 4 cách cấp thời tiết tương lai: chỉ quá khứ, thêm ECMWF HRES ngày đầu, thêm GEFS 7 ngày, thời tiết thật (cận trên, không dùng được khi vận hành).</li>
      <li><b>Nhìn lại 2 năm (D):</b> như C2 nhưng mô hình đọc 730 ngày thay vì 365 ngày.</li>
    </ul>
  </div>
  <div>
    <h3>Giải thích mô hình</h3>
    <p>Integrated Gradients (Sundararajan và cs., ICML 2017) chia con số dự báo thành phần đóng góp của từng đầu vào ở từng ngày. Trang hiện tỉ phần theo nguồn dữ liệu và độ dài "trí nhớ" (số ngày quá khứ giữ 90% đóng góp).</p>
  </div>
  <div>
    <h3>Nguồn và giấy phép</h3>
    <ul>
      <li>Dữ liệu và mã gốc: BiasCast, github.com/conestone/biascast; NeuralHydrology.</li>
      <li>Bản đồ nền © OpenStreetMap (ODbL), Protomaps; địa hình © Mapterhorn (mô hình độ cao mở của Áo, Thụy Sĩ, Đức, Séc và Copernicus, mapterhorn.com/attribution); thư viện MapLibre GL JS, Plotly.</li>
      <li>Mạng sông HydroRIVERS v1.0 (Lehner, B., Verdin, K., Jarvis, A. (2008), Eos 89(10): 93-94; Lehner, B., Grill, G. (2013), Hydrological Processes 27(15): 2171-2186.) <span class="legal">This product (River discharge forecasting, ticasslo.dev) incorporates data from the HydroSHEDS version 1 database which is © World Wildlife Fund, Inc. (2006-2022) and has been used herein under license. WWF has not evaluated the data as altered and incorporated within River discharge forecasting, and therefore gives no warranty regarding its accuracy, completeness, currency or suitability for any particular purpose. Portions of the HydroSHEDS v1 database incorporate data which are the intellectual property rights of © USGS (2006-2008), NASA (2000-2005), ESRI (1992-1998), CIAT (2004-2006), UNEP-WCMC (1993), WWF (2004), Commonwealth of Australia (2007), and Her Royal Majesty and the British Crown and are used under license. The HydroSHEDS v1 database and more information are available at https://www.hydrosheds.org.</span></li>
      <li>Dữ liệu dùng cho mục đích học tập, phi thương mại (CC BY-NC 4.0).</li>
    </ul>
  </div>
</div>`,
  en: `
<div class="cols">
  <div>
    <h3>The task</h3>
    <p>Discharge is the volume of water passing a river cross-section each second, in m³/s. This page forecasts the <b>daily peak discharge</b> (qmax) at each gauge and compares it with what was measured. The output is discharge, not water level or flood extent.</p>
    <p>This is a comparative study: same data, same parameter count (about 84,800, within 1%), the same 30-epoch training, swapping the architecture between LSTM, GRU, Transformer, S4D and Mamba (a State Space Model) to see which forecasts better.</p>
  </div>
  <div>
    <h3>Data</h3>
    <dl>
      <dt>Basins</dt><dd>{basins} basins of the daily Extended LamaH-CE dataset (Zenodo 17119635, CC BY-NC 4.0)</dd>
      <dt>Training</dt><dd>2003–2009</dd>
      <dt>Model selection</dt><dd>2010–2013 (validation)</dd>
      <dt>Test</dt><dd>2014–2017, shown on this page</dd>
      <dt>Inputs</dt><dd>ERA5-Land, E-OBS, MSWEP, GLEAM meteorology; ECMWF HRES forecasts; GEFSv12 forecasts (7-day task); recent discharge; basin attributes</dd>
    </dl>
  </div>
  <div>
    <h3>How to read it</h3>
    <ul>
      <li><b>Four flow levels</b> relative to the gauge itself in 1981–2013: low, medium, high, very high (top 2% of days).</li>
      <li><b>Return-period flood levels</b> of 1.5 / 2 / 5 / 10 years, fitted with a Gumbel distribution (L-moments) to annual peaks 1981–2013.</li>
      <li><b>NSE</b> (Nash–Sutcliffe): 1 is a perfect match, 0 equals guessing the mean. <b>KGE</b> (Kling–Gupta) combines correlation, bias and variability.</li>
      <li>Baselines: <b>BiasCast</b> (Konold et al., HESS 2026, the base paper of this project) and <b>same as today</b> (persistence: tomorrow equals today).</li>
    </ul>
  </div>
</div>
<div class="cols" style="margin-top:22px">
  <div>
    <h3>Three tasks</h3>
    <ul>
      <li><b>Tomorrow (C2):</b> 8 configurations of 5 architectures (same size or 3× larger), each trained 3 times; the page shows the run with the highest validation NSE.</li>
      <li><b>Next 7 days (H):</b> LSTM and Mamba with 4 ways of providing future weather: past only, first-day ECMWF HRES, 7-day GEFS, actual weather (upper bound, not available in operation).</li>
      <li><b>2-year memory (D):</b> like C2 but the model reads 730 days instead of 365.</li>
    </ul>
  </div>
  <div>
    <h3>Explaining the models</h3>
    <p>Integrated Gradients (Sundararajan et al., ICML 2017) splits a forecast into contributions from each input on each day. The page shows shares by data source and the model's "memory" (days of past data holding 90% of the contribution).</p>
  </div>
  <div>
    <h3>Sources and licences</h3>
    <ul>
      <li>Data and original code: BiasCast, github.com/conestone/biascast; NeuralHydrology.</li>
      <li>Basemap © OpenStreetMap (ODbL), Protomaps; terrain © Mapterhorn (open elevation models from Austria, Switzerland, Germany, Czechia and Copernicus, mapterhorn.com/attribution); MapLibre GL JS and Plotly libraries.</li>
      <li>River network HydroRIVERS v1.0 (Lehner, B., Verdin, K., Jarvis, A. (2008), Eos 89(10): 93-94; Lehner, B., Grill, G. (2013), Hydrological Processes 27(15): 2171-2186.) <span class="legal">This product (River discharge forecasting, ticasslo.dev) incorporates data from the HydroSHEDS version 1 database which is © World Wildlife Fund, Inc. (2006-2022) and has been used herein under license. WWF has not evaluated the data as altered and incorporated within River discharge forecasting, and therefore gives no warranty regarding its accuracy, completeness, currency or suitability for any particular purpose. Portions of the HydroSHEDS v1 database incorporate data which are the intellectual property rights of © USGS (2006-2008), NASA (2000-2005), ESRI (1992-1998), CIAT (2004-2006), UNEP-WCMC (1993), WWF (2004), Commonwealth of Australia (2007), and Her Royal Majesty and the British Crown and are used under license. The HydroSHEDS v1 database and more information are available at https://www.hydrosheds.org.</span></li>
      <li>Data used for non-commercial, educational purposes (CC BY-NC 4.0).</li>
    </ul>
  </div>
</div>`,
};

export function aboutHtml(lang, state) {
  return TEXT[lang].replace("{basins}", String(state.ids.length));
}
