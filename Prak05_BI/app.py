from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Coffee Shop Sales Intelligence | BI Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive polish
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #2D1E18 0%, #4A3328 50%, #6F4E37 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(45, 30, 24, 0.12);
    }
    .dashboard-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #FFF9F5 !important;
        letter-spacing: -0.5px;
    }
    .dashboard-header p {
        margin: 6px 0 0 0;
        font-size: 0.95rem;
        color: #E6D2C4;
    }
    .header-badges {
        margin-top: 14px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .header-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(4px);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #FFF0E6;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #EAE3DC;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(111, 78, 55, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(111, 78, 55, 0.1);
        border-color: #D4A373;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #79665B;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.65rem;
        font-weight: 700;
        color: #2D1E18;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .badge-positive {
        color: #2E7D32;
        background: #E8F5E9;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-negative {
        color: #C62828;
        background: #FFEBEE;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-neutral {
        color: #555555;
        background: #F5F5F5;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 500;
    }

    /* Key Insights Banner */
    .insight-callout {
        background: #FDF9F5;
        border-left: 4px solid #A67B5B;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 22px;
        border-top: 1px solid #F0E6DE;
        border-right: 1px solid #F0E6DE;
        border-bottom: 1px solid #F0E6DE;
    }
    .insight-callout-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #4A3328;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .insight-callout-content {
        font-size: 0.88rem;
        color: #5A4A42;
        line-height: 1.5;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background-color: #FBF9F7;
        border-right: 1px solid #EAE3DC;
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #EAE3DC;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 8px 8px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. DATA LOADING & CACHING
# ==========================================
folder = Path(__file__).resolve().parent
file_data = folder / "data_prepared" / "coffee_sales_prepared.csv"

if not file_data.exists():
    st.error("Data siap analisis belum ditemukan. Jalankan terlebih dahulu: `python audit_prepare_coffee.py`")
    st.stop()

@st.cache_data
def load_prepared_data(path_str, mtime):
    df = pd.read_csv(path_str)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"])
    return df

raw_df = load_prepared_data(str(file_data), file_data.stat().st_mtime_ns)

# Number formatting helpers
def fmt_id(v, digits=2):
    """Format standard Indonesian thousand '.' and decimal ',' representation."""
    if pd.isna(v) or v is None:
        return "Belum tersedia"
    return f"{v:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")

def fmt_int(v):
    if pd.isna(v) or v is None:
        return "0"
    return f"{int(v):,}".replace(",", ".")

# Color Palette Mappings
STORE_COLORS = {
    "Astoria": "#2B5B84",
    "Hell's Kitchen": "#D35400",
    "Lower Manhattan": "#27AE60",
}

CATEGORY_COLORS = {
    "Coffee": "#6F4E37",
    "Tea": "#2E7D32",
    "Bakery": "#D4A373",
    "Drinking Chocolate": "#795548",
    "Flavours": "#E67E22",
    "Coffee beans": "#3E2723",
    "Loose Tea": "#81C784",
    "Branded": "#5C6BC0",
    "Packaged Chocolate": "#8D6E63",
}

# ==========================================
# 3. SIDEBAR CONTROLS & INTERACTIVE FILTERS
# ==========================================
st.sidebar.markdown("### 🎛️ Filter Analisis")
st.sidebar.caption("Sesuaikan parameter analisis untuk memperdalam drilldown data.")

# 1. Filter Gerai
all_stores = sorted(raw_df["store_location"].dropna().unique().tolist())
select_all_stores = st.sidebar.checkbox("Pilih Semua Gerai", value=True)
if select_all_stores:
    selected_stores = st.sidebar.multiselect("Gerai Toko", all_stores, default=all_stores)
else:
    selected_stores = st.sidebar.multiselect("Gerai Toko", all_stores, default=[all_stores[0]])

if not selected_stores:
    st.sidebar.warning("Pilih minimal satu gerai.")
    st.stop()

# 2. Filter Periode / Bulan
all_months = sorted(raw_df["month"].dropna().unique().tolist())
month_preset = st.sidebar.radio(
    "Preset Periode",
    ["Semua Bulan (Jan–Jun)", "Q1 (Jan–Mar)", "Q2 (Apr–Jun)", "Bulan Terakhir (Juni)", "Kustom"],
    index=0,
    horizontal=False,
)

if month_preset == "Semua Bulan (Jan–Jun)":
    selected_months = all_months
elif month_preset == "Q1 (Jan–Mar)":
    selected_months = [m for m in all_months if m in ["2023-01", "2023-02", "2023-03"]]
elif month_preset == "Q2 (Apr–Jun)":
    selected_months = [m for m in all_months if m in ["2023-04", "2023-05", "2023-06"]]
elif month_preset == "Bulan Terakhir (Juni)":
    selected_months = ["2023-06"]
else:
    selected_months = st.sidebar.multiselect("Pilih Bulan", all_months, default=all_months)

if not selected_months:
    st.sidebar.warning("Pilih minimal satu bulan.")
    st.stop()

# 3. Filter Kategori Produk
all_cats = sorted(raw_df["product_category"].dropna().unique().tolist())
selected_cats = st.sidebar.multiselect("Kategori Produk", all_cats, default=all_cats)
if not selected_cats:
    st.sidebar.warning("Pilih minimal satu kategori produk.")
    st.stop()

# 4. Filter Tipe Produk (Dynamic based on selected categories)
avail_types = sorted(raw_df[raw_df["product_category"].isin(selected_cats)]["product_type"].dropna().unique().tolist())
with st.sidebar.expander("Filter Tipe Produk Lanjutan", expanded=False):
    select_all_types = st.checkbox("Pilih Semua Tipe Produk", value=True)
    if select_all_types:
        selected_types = avail_types
    else:
        selected_types = st.multiselect("Pilih Tipe Spesifik", avail_types, default=avail_types)
    if not selected_types:
        st.warning("Pilih minimal satu tipe produk.")
        st.stop()

# 5. Filter Hari dalam Seminggu
day_order_id = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
day_preset = st.sidebar.selectbox("Filter Hari", ["Semua Hari (7 Hari)", "Hari Kerja (Senin–Jumat)", "Akhir Pekan (Sabtu–Minggu)", "Pilih Hari Tertentu"])
if day_preset == "Semua Hari (7 Hari)":
    selected_days = day_order_id
elif day_preset == "Hari Kerja (Senin–Jumat)":
    selected_days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]
elif day_preset == "Akhir Pekan (Sabtu–Minggu)":
    selected_days = ["Sabtu", "Minggu"]
else:
    selected_days = st.sidebar.multiselect("Pilih Hari", day_order_id, default=day_order_id)
    if not selected_days:
        st.sidebar.warning("Pilih minimal satu hari.")
        st.stop()

# 6. Filter Rentang Jam
min_hour = int(raw_df["hour"].min())
max_hour = int(raw_df["hour"].max())
selected_hour_range = st.sidebar.slider(
    "Rentang Jam Transaksi",
    min_value=min_hour,
    max_value=max_hour,
    value=(min_hour, max_hour),
    format="%d:00",
)

# Apply All Filters to Dataset
filtered_df = raw_df[
    (raw_df["store_location"].isin(selected_stores))
    & (raw_df["month"].isin(selected_months))
    & (raw_df["product_category"].isin(selected_cats))
    & (raw_df["product_type"].isin(selected_types))
    & (raw_df["day_name_id"].isin(selected_days))
    & (raw_df["hour"].between(selected_hour_range[0], selected_hour_range[1]))
].copy()

if filtered_df.empty:
    st.error("Tidak ada data transaksi yang memenuhi kriteria filter terpilih. Harap sesuaikan filter Anda.")
    st.stop()

# Sidebar Data Volume Card
filter_pct = (len(filtered_df) / len(raw_df)) * 100
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="background: #FFFFFF; border: 1px solid #EAE3DC; border-radius: 10px; padding: 12px; font-size: 0.82rem;">
        <strong style="color: #4A3328;">📊 Status Data Terfilter:</strong><br>
        <span style="font-size: 1.15rem; font-weight: 700; color: #2D1E18;">{fmt_int(len(filtered_df))}</span> / {fmt_int(len(raw_df))} baris<br>
        <span style="color: #79665B;">Mencakup <b>{filter_pct:.1f}%</b> dari keseluruhan dataset.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 4. HEADER & TOP EXECUTIVE KPI METRICS
# ==========================================
date_start_str = filtered_df["transaction_date"].min().strftime("%d %b %Y")
date_end_str = filtered_df["transaction_date"].max().strftime("%d %b %Y")
stores_str = ", ".join(selected_stores) if len(selected_stores) <= 2 else f"{len(selected_stores)} Gerai Terpilih"

st.markdown(
    f"""
    <div class="dashboard-header">
        <h1>☕ Coffee Shop Sales Intelligence Dashboard</h1>
        <p>Analisis Performa Bisnis, Pola Transaksi, Komposisi Produk, dan Waktu Operasional</p>
        <div class="header-badges">
            <span class="header-badge">📅 Periode: {date_start_str} – {date_end_str}</span>
            <span class="header-badge">🏪 Gerai: {stores_str}</span>
            <span class="header-badge">📦 Total Catatan: {fmt_int(len(filtered_df))} baris</span>
            <span class="header-badge">🕒 Jam: {selected_hour_range[0]:02d}:00 – {selected_hour_range[1]:02d}:00</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compute Global KPIs for active selection
total_sales = filtered_df["sales"].sum()
total_transactions = filtered_df["transaction_id"].nunique()
total_units = filtered_df["transaction_qty"].sum()
avg_transaction_val = total_sales / total_transactions if total_transactions else 0.0
active_days = filtered_df["transaction_date"].dt.normalize().nunique()
sales_per_day = total_sales / active_days if active_days else 0.0
units_per_trans = total_units / total_transactions if total_transactions else 0.0

# Calculate MoM growth for the last two available months in the selection
monthly_calc = (
    filtered_df.groupby("month")
    .agg(sales=("sales", "sum"), trans=("transaction_id", "nunique"))
    .sort_index()
)
if len(monthly_calc) >= 2:
    last_m = monthly_calc.index[-1]
    prev_m = monthly_calc.index[-2]
    last_sales = monthly_calc.loc[last_m, "sales"]
    prev_sales = monthly_calc.loc[prev_m, "sales"]
    mom_sales_growth = ((last_sales - prev_sales) / prev_sales) * 100 if prev_sales else None
    growth_badge = (
        f'<span class="badge-positive">+{mom_sales_growth:.2f}% MoM ({last_m})</span>'
        if mom_sales_growth >= 0
        else f'<span class="badge-negative">{mom_sales_growth:.2f}% MoM ({last_m})</span>'
    )
else:
    growth_badge = '<span class="badge-neutral">Periode tunggal / kustom</span>'

# Render 5 KPI Cards in columns
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Penjualan</div>
            <div class="metric-val">{fmt_id(total_sales)}</div>
            <div class="metric-sub">{growth_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Transaksi Tercatat</div>
            <div class="metric-val">{fmt_int(total_transactions)}</div>
            <div class="metric-sub"><span class="badge-neutral">Grain: 1 baris / transaksi</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Unit Terjual</div>
            <div class="metric-val">{fmt_int(total_units)}</div>
            <div class="metric-sub"><span class="badge-neutral">{units_per_trans:.2f} unit/transaksi</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Transaksi (ATV)</div>
            <div class="metric-val">{fmt_id(avg_transaction_val)}</div>
            <div class="metric-sub"><span class="badge-neutral">Penjualan ÷ Transaksi</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Penjualan / Hari</div>
            <div class="metric-val">{fmt_id(sales_per_day)}</div>
            <div class="metric-sub"><span class="badge-neutral">{active_days} hari operasional aktif</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# Dynamic Automated Executive Insights
top_store_series = filtered_df.groupby("store_location")["sales"].sum().sort_values(ascending=False)
top_store_name = top_store_series.index[0]
top_store_share = (top_store_series.iloc[0] / total_sales) * 100

top_cat_series = filtered_df.groupby("product_category")["sales"].sum().sort_values(ascending=False)
top_cat_name = top_cat_series.index[0]
top_cat_share = (top_cat_series.iloc[0] / total_sales) * 100

peak_hour_val = filtered_df.groupby("hour")["transaction_id"].nunique().idxmax()
peak_hour_trans = filtered_df.groupby("hour")["transaction_id"].nunique().max()

peak_day_val = filtered_df.groupby("day_name_id")["transaction_id"].nunique().idxmax()

st.markdown(
    f"""
    <div class="insight-callout">
        <div class="insight-callout-title">💡 Ringkasan Temuan Cepat (Otomatis berdasarkan Filter Aktif)</div>
        <div class="insight-callout-content">
            • <b>Gerai Terdepan:</b> <code>{top_store_name}</code> berkontribusi tertinggi dengan total penjualan <b>{fmt_id(top_store_series.iloc[0])}</b> ({top_store_share:.1f}% dari total terpilih).<br>
            • <b>Kategori Unggulan:</b> <code>{top_cat_name}</code> memimpin perolehan penjualan dengan pangsa <b>{top_cat_share:.1f}%</b> ({fmt_id(top_cat_series.iloc[0])}).<br>
            • <b>Waktu Tersibuk:</b> Puncak volume transaksi tercatat pada pukul <b>{peak_hour_val:02d}:00</b> ({fmt_int(peak_hour_trans)} transaksi), dengan hari paling ramai jatuh pada <b>{peak_day_val}</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 5. MULTI-TAB DRILLDOWN ARCHITECTURE
# ==========================================
tab_overview, tab_store, tab_product, tab_time, tab_explorer, tab_method = st.tabs([
    "📊 Ringkasan Eksekutif",
    "🏪 Analisis Gerai & Komparasi",
    "☕ Drilldown Kategori & Produk",
    "⏰ Pola Waktu & Operasional",
    "🔍 Eksplorasi Data & Ekspor",
    "📋 Metodologi & Audit",
])

# ----------------------------------------------------
# TAB 1: RINGKASAN EKSEKUTIF
# ----------------------------------------------------
with tab_overview:
    st.subheader("Tren Penjualan & Kinerja Bulanan")
    
    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        metric_choice = st.radio(
            "Tampilkan Metrik Tren:",
            ["Nilai Penjualan", "Transaksi Tercatat", "Unit Terjual", "Rata-rata Nilai Transaksi"],
            horizontal=True,
        )
    with t_col2:
        granularity = st.selectbox("Agregasi Waktu Tren:", ["Bulanan", "Mingguan", "Harian"], index=0)

    # Trend preparation
    if granularity == "Bulanan":
        trend_df = (
            filtered_df.groupby("month")
            .agg(
                sales=("sales", "sum"),
                transactions=("transaction_id", "nunique"),
                units=("transaction_qty", "sum"),
            )
            .reset_index()
            .rename(columns={"month": "periode"})
        )
    elif granularity == "Mingguan":
        df_week = filtered_df.copy()
        df_week["periode"] = df_week["transaction_date"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
        trend_df = (
            df_week.groupby("periode")
            .agg(
                sales=("sales", "sum"),
                transactions=("transaction_id", "nunique"),
                units=("transaction_qty", "sum"),
            )
            .reset_index()
        )
    else:
        df_day = filtered_df.copy()
        df_day["periode"] = df_day["transaction_date"].dt.strftime("%Y-%m-%d")
        trend_df = (
            df_day.groupby("periode")
            .agg(
                sales=("sales", "sum"),
                transactions=("transaction_id", "nunique"),
                units=("transaction_qty", "sum"),
            )
            .reset_index()
        )

    trend_df["average_transaction_value"] = trend_df["sales"] / trend_df["transactions"]

    # Select target column
    col_map = {
        "Nilai Penjualan": ("sales", "Nilai Penjualan"),
        "Transaksi Tercatat": ("transactions", "Jumlah Transaksi"),
        "Unit Terjual": ("units", "Unit Terjual"),
        "Rata-rata Nilai Transaksi": ("average_transaction_value", "ATV (Penjualan / Transaksi)"),
    }
    y_col, y_label = col_map[metric_choice]

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=trend_df["periode"],
            y=trend_df[y_col],
            mode="lines+markers",
            name=y_label,
            line=dict(color="#6F4E37", width=3),
            marker=dict(size=8, color="#D4A373", line=dict(color="#4A3328", width=1.5)),
            hovertemplate="Periode: %{x}<br>" + y_label + ": %{y:,.2f}<extra></extra>",
        )
    )

    # Average benchmark line
    mean_val = trend_df[y_col].mean()
    fig_trend.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="#A67B5B",
        annotation_text=f"Rata-rata: {mean_val:,.2f}",
        annotation_position="bottom right",
    )

    fig_trend.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis_title="Periode Waktu",
        yaxis_title=y_label,
        yaxis_rangemode="tozero",
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Two columns: Store Contribution & Category Contribution
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Penjualan Berdasarkan Gerai")
        by_store = (
            filtered_df.groupby("store_location", as_index=False)
            .agg(sales=("sales", "sum"), trans=("transaction_id", "nunique"), units=("transaction_qty", "sum"))
            .sort_values("sales", ascending=False)
        )
        by_store["share"] = (by_store["sales"] / total_sales) * 100
        by_store["color"] = by_store["store_location"].map(STORE_COLORS).fillna("#6F4E37")

        fig_store = go.Figure()
        fig_store.add_trace(
            go.Bar(
                x=by_store["store_location"],
                y=by_store["sales"],
                text=by_store.apply(lambda r: f"{fmt_id(r['sales'])}<br>({r['share']:.1f}%)", axis=1),
                textposition="outside",
                marker_color=by_store["color"],
                hovertemplate="<b>%{x}</b><br>Penjualan: %{y:,.2f}<extra></extra>",
            )
        )
        fig_store.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="Lokasi Gerai",
            yaxis_title="Nilai Penjualan",
            yaxis_rangemode="tozero",
        )
        st.plotly_chart(fig_store, use_container_width=True)

    with c_right:
        st.subheader("Kontribusi Kategori Produk")
        by_cat = (
            filtered_df.groupby("product_category", as_index=False)
            .agg(sales=("sales", "sum"), trans=("transaction_id", "nunique"))
            .sort_values("sales", ascending=False)
        )
        fig_cat = px.pie(
            by_cat,
            names="product_category",
            values="sales",
            hole=0.55,
            color="product_category",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig_cat.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Penjualan: %{value:,.2f}<br>Pangsa: %{percent}<extra></extra>",
        )
        fig_cat.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Monthly Summary Table expander
    with st.expander("📋 Lihat Tabel Ringkasan Bulanan Lengkap", expanded=False):
        months_order = sorted(filtered_df["month"].unique().tolist())
        monthly_table = (
            filtered_df.groupby("month")
            .agg(
                penjualan=("sales", "sum"),
                transaksi=("transaction_id", "nunique"),
                unit=("transaction_qty", "sum"),
            )
            .reindex(months_order)
            .reset_index()
        )
        monthly_table["prev_penjualan"] = monthly_table["penjualan"].shift(1)
        monthly_table["pertumbuhan_pct"] = (
            (monthly_table["penjualan"] - monthly_table["prev_penjualan"]) / monthly_table["prev_penjualan"] * 100
        )
        monthly_table["rata_rata_transaksi"] = monthly_table["penjualan"] / monthly_table["transaksi"]

        # Formatted display dataframe
        display_month = pd.DataFrame({
            "Bulan": monthly_table["month"],
            "Nilai Penjualan": monthly_table["penjualan"].apply(lambda v: fmt_id(v)),
            "Transaksi Tercatat": monthly_table["transaksi"].apply(lambda v: fmt_int(v)),
            "Unit Terjual": monthly_table["unit"].apply(lambda v: fmt_int(v)),
            "Pertumbuhan MoM (%)": monthly_table["pertumbuhan_pct"].apply(
                lambda v: f"{v:+.2f}%" if pd.notna(v) else "Belum tersedia"
            ),
            "Rata-rata Nilai Transaksi": monthly_table["rata_rata_transaksi"].apply(lambda v: fmt_id(v)),
        })
        st.dataframe(display_month, hide_index=True, use_container_width=True)

# ----------------------------------------------------
# TAB 2: ANALISIS GERAI & KOMPARASI
# ----------------------------------------------------
with tab_store:
    st.subheader("🏪 Evaluasi Kinerja & Komparasi Antargerai")
    st.caption("Perbandingan performa penjualan, lintasan tren bulanan, dan komposisi produk pada masing-masing gerai.")

    # Side-by-side store metric cards
    store_summary = (
        filtered_df.groupby("store_location")
        .agg(
            total_sales=("sales", "sum"),
            total_trans=("transaction_id", "nunique"),
            total_units=("transaction_qty", "sum"),
            days=("transaction_date", lambda d: d.dt.normalize().nunique()),
        )
        .reset_index()
    )
    store_summary["atv"] = store_summary["total_sales"] / store_summary["total_trans"]
    store_summary["sales_per_day"] = store_summary["total_sales"] / store_summary["days"]
    store_summary["share"] = (store_summary["total_sales"] / total_sales) * 100

    s_cols = st.columns(len(store_summary))
    for idx, row in store_summary.iterrows():
        s_name = row["store_location"]
        s_color = STORE_COLORS.get(s_name, "#6F4E37")
        with s_cols[idx]:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border-top: 4px solid {s_color}; border-radius: 12px; padding: 16px; border-left: 1px solid #EAE3DC; border-right: 1px solid #EAE3DC; border-bottom: 1px solid #EAE3DC; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
                    <div style="font-size: 1.1rem; font-weight: 700; color: {s_color}; margin-bottom: 8px;">📍 {s_name}</div>
                    <div style="font-size: 0.8rem; color: #79665B;">Total Penjualan:</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #2D1E18;">{fmt_id(row['total_sales'])}</div>
                    <div style="font-size: 0.8rem; color: #888888; margin-top: 2px;">Pangsa: <b>{row['share']:.1f}%</b> dari total</div>
                    <hr style="margin: 8px 0; border: none; border-top: 1px dashed #EAE3DC;">
                    <div style="font-size: 0.82rem; color: #555555; line-height: 1.6;">
                        • Transaksi: <b>{fmt_int(row['total_trans'])}</b><br>
                        • Unit Terjual: <b>{fmt_int(row['total_units'])}</b><br>
                        • Rata-rata/Transaksi (ATV): <b>{fmt_id(row['atv'])}</b><br>
                        • Penjualan/Hari: <b>{fmt_id(row['sales_per_day'])}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Store Monthly Trend Lines Comparison
    st.subheader("Tren Penjualan Bulanan Antargerai")
    store_month_df = (
        filtered_df.groupby(["month", "store_location"])
        .agg(sales=("sales", "sum"), trans=("transaction_id", "nunique"))
        .reset_index()
    )
    fig_store_trend = px.line(
        store_month_df,
        x="month",
        y="sales",
        color="store_location",
        markers=True,
        color_discrete_map=STORE_COLORS,
        labels={"month": "Bulan", "sales": "Nilai Penjualan", "store_location": "Gerai"},
    )
    fig_store_trend.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=30, b=30),
        yaxis_rangemode="tozero",
        hovermode="x unified",
    )
    fig_store_trend.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(fig_store_trend, use_container_width=True)

    # Store Category Mix Drilldown (Absolute vs 100% Normalized)
    st.subheader("Komposisi Kategori per Gerai")
    mix_mode = st.radio("Mode Tampilan Komposisi:", ["Nilai Penjualan Absolut", "Proporsi Persentase (100% Normalized)"], horizontal=True)
    
    store_cat = (
        filtered_df.groupby(["store_location", "product_category"], as_index=False)
        .agg(sales=("sales", "sum"))
    )

    if mix_mode == "Proporsi Persentase (100% Normalized)":
        # Calculate percentage within each store
        store_totals = store_cat.groupby("store_location")["sales"].transform("sum")
        store_cat["pct"] = (store_cat["sales"] / store_totals) * 100
        fig_mix = px.bar(
            store_cat,
            x="store_location",
            y="pct",
            color="product_category",
            color_discrete_map=CATEGORY_COLORS,
            labels={"store_location": "Gerai", "pct": "Persentase (%)", "product_category": "Kategori"},
        )
        fig_mix.update_layout(
            barmode="stack",
            yaxis_ticksuffix="%",
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=30, b=30),
        )
        fig_mix.update_traces(hovertemplate="<b>%{x}</b><br>Kategori: %{data.name}<br>Porsi: %{y:.1f}%<extra></extra>")
    else:
        fig_mix = px.bar(
            store_cat,
            x="store_location",
            y="sales",
            color="product_category",
            barmode="group",
            color_discrete_map=CATEGORY_COLORS,
            labels={"store_location": "Gerai", "sales": "Nilai Penjualan", "product_category": "Kategori"},
        )
        fig_mix.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=30, b=30),
        )
        fig_mix.update_traces(hovertemplate="<b>%{x}</b><br>Kategori: %{data.name}<br>Penjualan: %{y:,.2f}<extra></extra>")
    
    st.plotly_chart(fig_mix, use_container_width=True)

# ----------------------------------------------------
# TAB 3: DRILLDOWN KATEGORI & PRODUK
# ----------------------------------------------------
with tab_product:
    st.subheader("☕ Hierarki Produk & Analisis Penjualan")
    st.caption("Eksplorasi berjenjang dari Kategori Produk → Tipe Produk → Rincian Produk Spesifik.")

    # Hierarchical Treemap
    st.markdown("#### Peta Proporsi Hierarki (Treemap: Kategori → Tipe Produk)")
    tree_data = (
        filtered_df.groupby(["product_category", "product_type"], as_index=False)
        .agg(sales=("sales", "sum"), units=("transaction_qty", "sum"))
    )
    fig_tree = px.treemap(
        tree_data,
        path=["product_category", "product_type"],
        values="sales",
        color="product_category",
        color_discrete_map=CATEGORY_COLORS,
    )
    fig_tree.update_traces(
        hovertemplate="<b>%{label}</b><br>Penjualan: %{value:,.2f}<extra></extra>",
        textinfo="label+value+percent parent",
    )
    fig_tree.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")
    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.subheader("🏆 Peringkat Produk Teratas & Terbawah")
        rank_mode = st.radio(
            "Tipe Peringkat:",
            ["Top Produk Terlaris", "Produk dengan Penjualan Terendah (Slow Movers)"],
            horizontal=True,
        )
        n_items = st.selectbox("Jumlah Item:", [5, 10, 15, 20], index=1)
        rank_metric = st.selectbox("Berdasarkan:", ["Nilai Penjualan", "Unit Terjual"])

        metric_target = "sales" if rank_metric == "Nilai Penjualan" else "units"
        product_rank = (
            filtered_df.groupby(["product_id", "product_detail", "product_category"], as_index=False)
            .agg(sales=("sales", "sum"), units=("transaction_qty", "sum"))
        )

        if rank_mode == "Top Produk Terlaris":
            chart_data = product_rank.nlargest(n_items, metric_target).sort_values(metric_target, ascending=True)
            bar_color = "#6F4E37"
        else:
            chart_data = product_rank.nsmallest(n_items, metric_target).sort_values(metric_target, ascending=False)
            bar_color = "#A67B5B"

        chart_data["label"] = chart_data["product_detail"] + " (" + chart_data["product_category"] + ")"
        fig_prod = px.bar(
            chart_data,
            x=metric_target,
            y="label",
            orientation="h",
            color_discrete_sequence=[bar_color],
            labels={metric_target: rank_metric, "label": "Produk"},
        )
        fig_prod.update_layout(
            template="plotly_white",
            height=440,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_prod.update_traces(hovertemplate="<b>%{y}</b><br>" + rank_metric + ": %{x:,.2f}<extra></extra>")
        st.plotly_chart(fig_prod, use_container_width=True)

    with p_col2:
        st.subheader("🏷️ Analisis Harga Satuan per Kategori")
        st.caption("Distribusi rata-rata harga satuan (unit_price) pada tiap kelompok produk.")
        price_summary = (
            filtered_df.groupby("product_category")
            .agg(
                avg_price=("unit_price", "mean"),
                min_price=("unit_price", "min"),
                max_price=("unit_price", "max"),
                total_units=("transaction_qty", "sum"),
            )
            .reset_index()
            .sort_values("avg_price", ascending=True)
        )
        fig_price = px.bar(
            price_summary,
            x="avg_price",
            y="product_category",
            orientation="h",
            color="product_category",
            color_discrete_map=CATEGORY_COLORS,
            labels={"avg_price": "Rata-rata Harga Satuan", "product_category": "Kategori"},
        )
        fig_price.update_layout(
            template="plotly_white",
            height=440,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_price.update_traces(
            hovertemplate="<b>%{y}</b><br>Rata-rata Harga: %{x:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_price, use_container_width=True)

# ----------------------------------------------------
# TAB 4: POLA WAKTU & OPERASIONAL
# ----------------------------------------------------
with tab_time:
    st.subheader("⏰ Analisis Pola Waktu & Kepadatan Transaksi")
    st.caption("Mengidentifikasi jendela jam sibuk dan hari dengan intensitas transaksi tertinggi untuk perencanaan operasional.")

    # Heatmap: Hari x Jam
    st.markdown("#### Peta Panas (Heatmap) Kepadatan Transaksi: Hari vs Jam")
    heat_raw = (
        filtered_df.groupby(["day_name_id", "hour"])
        .agg(trans=("transaction_id", "nunique"))
        .reset_index()
    )
    heat_pivot = (
        heat_raw.pivot(index="day_name_id", columns="hour", values="trans")
        .reindex(day_order_id)
        .fillna(0)
    )

    fig_heat = px.imshow(
        heat_pivot,
        labels=dict(x="Jam Transaksi", y="Hari", color="Jumlah Transaksi"),
        x=[f"{h:02d}:00" for h in heat_pivot.columns],
        y=heat_pivot.index,
        color_continuous_scale="YlOrBr",
        aspect="auto",
    )
    fig_heat.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=30),
    )
    fig_heat.update_traces(hovertemplate="Hari: %{y}<br>Jam: %{x}<br>Transaksi: %{z:,}<extra></extra>")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Two columns: Hourly Breakdown & Day of Week Breakdown
    t_left, t_right = st.columns(2)
    with t_left:
        st.subheader("Distribusi Transaksi per Jam")
        hourly = (
            filtered_df.groupby("hour", as_index=False)
            .agg(transactions=("transaction_id", "nunique"), sales=("sales", "sum"))
            .sort_values("hour")
        )
        hourly["label"] = hourly["hour"].apply(lambda h: f"{h:02d}:00")
        peak_idx = hourly["transactions"].idxmax()
        hourly_colors = ["#D35400" if i == peak_idx else "#8D6E63" for i in hourly.index]

        fig_hour = go.Figure()
        fig_hour.add_trace(
            go.Bar(
                x=hourly["label"],
                y=hourly["transactions"],
                marker_color=hourly_colors,
                hovertemplate="<b>Jam %{x}</b><br>Transaksi: %{y:,}<extra></extra>",
            )
        )
        fig_hour.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_title="Jam",
            yaxis_title="Transaksi Tercatat",
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        st.caption(f"📌 Puncak aktivitas transaksi harian tercatat pada pukul **{hourly.loc[peak_idx, 'label']}**.")

    with t_right:
        st.subheader("Distribusi Transaksi menurut Hari")
        daily = (
            filtered_df.groupby("day_name_id", as_index=False)
            .agg(transactions=("transaction_id", "nunique"), sales=("sales", "sum"))
        )
        # reorder by standard Monday-Sunday
        daily["sort_order"] = daily["day_name_id"].map({d: i for i, d in enumerate(day_order_id)})
        daily = daily.sort_values("sort_order")

        fig_day = px.bar(
            daily,
            x="day_name_id",
            y="transactions",
            color="day_name_id",
            color_discrete_sequence=["#6F4E37"] * 7,
            labels={"day_name_id": "Hari", "transactions": "Transaksi Tercatat"},
        )
        fig_day.update_layout(
            template="plotly_white",
            height=340,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_title="Hari",
            yaxis_title="Transaksi Tercatat",
        )
        fig_day.update_traces(hovertemplate="<b>%{x}</b><br>Transaksi: %{y:,}<extra></extra>")
        st.plotly_chart(fig_day, use_container_width=True)

    # Time Segment Breakdown (Pagi, Siang, Sore, Malam)
    st.markdown("#### Segmentasi Sesi Operasional Toko")
    def categorize_session(h):
        if h <= 10:
            return "Pagi (06:00 – 10:59)"
        elif h <= 14:
            return "Siang (11:00 – 14:59)"
        elif h <= 18:
            return "Sore (15:00 – 18:59)"
        else:
            return "Malam (19:00 – 20:59)"

    df_session = filtered_df.copy()
    df_session["session"] = df_session["hour"].apply(categorize_session)
    session_agg = (
        df_session.groupby("session", as_index=False)
        .agg(sales=("sales", "sum"), trans=("transaction_id", "nunique"), units=("transaction_qty", "sum"))
    )
    session_agg["atv"] = session_agg["sales"] / session_agg["trans"]
    session_agg["share"] = (session_agg["sales"] / total_sales) * 100

    col_sess = st.columns(len(session_agg))
    for idx, row in session_agg.iterrows():
        with col_sess[idx]:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #EAE3DC; border-radius: 10px; padding: 14px; text-align: center;">
                    <strong style="color: #4A3328;">{row['session']}</strong><br>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #2D1E18;">{fmt_id(row['sales'])}</span><br>
                    <span style="font-size: 0.8rem; color: #79665B;">Porsi: <b>{row['share']:.1f}%</b> ({fmt_int(row['trans'])} trx)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------
# TAB 5: EKSPLORASI DATA & EKSPOR
# ----------------------------------------------------
with tab_explorer:
    st.subheader("🔍 Eksplorasi Data & Ekspor Hasil Agregasi")
    st.caption("Pilih level ringkasan untuk meneliti performa detail atau mengunduh dataset untuk keperluan pelaporan.")

    agg_view = st.selectbox(
        "Pilih Sudut Pandang Data:",
        [
            "Ringkasan per Produk Detail",
            "Ringkasan per Gerai & Kategori",
            "Ringkasan per Gerai & Bulan",
            "Sampel Catatan Transaksi Mentah (Maks. 1.000 baris)",
        ],
    )

    if agg_view == "Ringkasan per Produk Detail":
        search_kw = st.text_input("Cari nama produk (misal: Latte, Dark Chocolate, Scone):", "")
        export_df = (
            filtered_df.groupby(
                ["product_id", "product_category", "product_type", "product_detail"],
                as_index=False,
            )
            .agg(
                penjualan=("sales", "sum"),
                unit_terjual=("transaction_qty", "sum"),
                transaksi=("transaction_id", "nunique"),
                rata_harga=("unit_price", "mean"),
            )
            .sort_values("penjualan", ascending=False)
        )
        if search_kw.strip():
            export_df = export_df[export_df["product_detail"].str.contains(search_kw.strip(), case=False, na=False)]

        st.dataframe(
            export_df.rename(columns={
                "product_id": "ID Produk",
                "product_category": "Kategori",
                "product_type": "Tipe Produk",
                "product_detail": "Detail Produk",
                "penjualan": "Total Penjualan",
                "unit_terjual": "Unit Terjual",
                "transaksi": "Transaksi Tercatat",
                "rata_harga": "Rata-rata Harga",
            }),
            use_container_width=True,
            hide_index=True,
        )

    elif agg_view == "Ringkasan per Gerai & Kategori":
        export_df = (
            filtered_df.groupby(["store_location", "product_category"], as_index=False)
            .agg(
                penjualan=("sales", "sum"),
                transaksi=("transaction_id", "nunique"),
                unit=("transaction_qty", "sum"),
            )
            .sort_values(["store_location", "penjualan"], ascending=[True, False])
        )
        export_df["rata_transaksi"] = export_df["penjualan"] / export_df["transaksi"]
        st.dataframe(
            export_df.rename(columns={
                "store_location": "Gerai",
                "product_category": "Kategori",
                "penjualan": "Total Penjualan",
                "transaksi": "Transaksi",
                "unit": "Unit Terjual",
                "rata_transaksi": "Rata-rata Nilai Transaksi",
            }),
            use_container_width=True,
            hide_index=True,
        )

    elif agg_view == "Ringkasan per Gerai & Bulan":
        export_df = (
            filtered_df.groupby(["store_location", "month"], as_index=False)
            .agg(
                penjualan=("sales", "sum"),
                transaksi=("transaction_id", "nunique"),
                unit=("transaction_qty", "sum"),
            )
            .sort_values(["store_location", "month"])
        )
        export_df["rata_transaksi"] = export_df["penjualan"] / export_df["transaksi"]
        st.dataframe(
            export_df.rename(columns={
                "store_location": "Gerai",
                "month": "Bulan",
                "penjualan": "Total Penjualan",
                "transaksi": "Transaksi",
                "unit": "Unit Terjual",
                "rata_transaksi": "Rata-rata Nilai Transaksi",
            }),
            use_container_width=True,
            hide_index=True,
        )

    else:
        sample_rows = min(1000, len(filtered_df))
        export_df = filtered_df.head(sample_rows).copy()
        st.caption(f"Menampilkan {sample_rows:,} baris pertama dari filter aktif.")
        st.dataframe(export_df, use_container_width=True, hide_index=True)

    # CSV Download Button
    csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Unduh Data Tampilan Ini (.CSV)",
        data=csv_bytes,
        file_name=f"coffee_shop_export_{agg_view.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

# ----------------------------------------------------
# TAB 6: METODOLOGI & AUDIT
# ----------------------------------------------------
with tab_method:
    st.subheader("📋 Definisi Indikator & Metodologi Analisis")
    st.markdown(
        """
        Dashboard ini dibangun dengan mematuhi standar integritas audit data bisnis. 
        Berikut rincian definisi matematis, cakupan filter, dan catatan metodologi yang mendasari setiap indikator.
        """
    )

    indicator_def = pd.DataFrame([
        [
            "Total Penjualan",
            "Σ (transaction_qty × unit_price)",
            "Seluruh transaksi yang lolos kriteria filter aktif",
            "Satuan mata uang tidak dinyatakan secara eksplisit pada sumber data Excel.",
        ],
        [
            "Transaksi Tercatat",
            "Jumlah transaction_id unik",
            "Sesuai filter gerai, periode, dan waktu aktif",
            "Merujuk pada transaksi tercatat pada grain data (satu produk/transaksi), bukan jumlah pelanggan unik.",
        ],
        [
            "Rata-rata Nilai Transaksi (ATV)",
            "Total Penjualan ÷ Transaksi Tercatat",
            "Dihitung langsung dari total penjualan dibagi total transaksi tercatat pada segmen terpilih",
            "Tidak boleh dirata-ratakan dari rata-rata bulanan atau rata-rata antargerai (Simpson's Paradox).",
        ],
        [
            "Pertumbuhan Terakhir (MoM)",
            "(Penjualan Bulan Terakhir − Bulan Sebelumnya) ÷ Bulan Sebelumnya × 100%",
            "Membandingkan bulan terakhir dalam filter dengan bulan tepat sebelumnya pada gerai yang sama",
            "Hanya dihitung jika terdapat minimal 2 bulan berturutan pada filter aktif.",
        ],
        [
            "Unit Terjual",
            "Σ (transaction_qty)",
            "Akumulasi kuantitas produk yang dipesan",
            "Mengukur volume fisik barang yang keluar dari gerai.",
        ],
    ], columns=["Indikator", "Rumus Matematis", "Cakupan Analisis", "Catatan Interpretasi"])

    st.dataframe(indicator_def, hide_index=True, use_container_width=True)

    st.markdown("#### 🛡️ Laporan Audit Kualitas Data")
    st.markdown(
        """
        - **Integritas Baris:** Seluruh **149.116 baris transaksi asli dipertahankan** tanpa pemotongan atau penghapusan otomatis.
        - **Rekonsiliasi Nilai:** Total penjualan hasil perkalian unit dan harga telah direkonsiliasi secara presisi sebesar **698.812,33**.
        - **Variasi Harga Produk:** Sebanyak 15 produk memiliki variasi harga satuan (misalnya perbedaan ukuran porsi atau kebijakan harga) dan dipertahankan apa adanya sesuai transaksi nyata.
        - **Batas Interpretasi:** Data transaksi kasir tidak memuat biaya operasional (HPP), data persediaan/stok, identitas pelanggan, maupun durasi antrean layanan. Karena itu, volume penjualan tidak dapat secara langsung disamakan dengan laba bersih.
        """
    )
