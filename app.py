# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO - DISEÑO MEJORADO
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time
import requests

# Configuración global
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Aplicar estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #64B5F6;
    }
    .metric-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #1976D2;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #0D47A1;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .ticker-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #2196F3;
    }
    .positive-value {
        color: #2E7D32;
        font-weight: 600;
    }
    .negative-value {
        color: #C62828;
        font-weight: 600;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Parámetros editables
Rf = 0.0435   # riesgo libre
Rm = 0.085    # retorno mercado
Tc0 = 0.21    # tasa impositiva por defecto

# Orden de sectores
SECTOR_RANK = {
    "Consumer Defensive": 1,
    "Consumer Cyclical": 2,
    "Healthcare": 3,
    "Technology": 4,
    "Financial Services": 5,
    "Industrials": 6,
    "Communication Services": 7,
    "Energy": 8,
    "Real Estate": 9,
    "Utilities": 10,
    "Basic Materials": 11,
    "Unknown": 99,
}

MAX_TICKERS_PER_CHART = 10

# =============================================================
# FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
    return obj.iloc[0] if hasattr(obj, "iloc") and not obj.empty else obj

def seek_row(df, keys):
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series([0], index=df.columns[:1])

def format_number(x, decimals=2, is_percent=False):
    if pd.isna(x):
        return "N/D"
    if is_percent:
        return f"{x*100:.{decimals}f}%"
    return f"{x:.{decimals}f}"

def calc_ke(beta):
    return Rf + beta * (Rm - Rf)

def calc_kd(interest, debt):
    return interest / debt if debt else 0

def calc_wacc(mcap, debt, ke, kd, t):
    total = (mcap or 0) + (debt or 0)
    return (mcap/total)*ke + (debt/total)*kd*(1-t) if total else None

def cagr4(fin, metric):
    if metric not in fin.index:
        return None
    v = fin.loc[metric].dropna().iloc[:4]
    return (v.iloc[0]/v.iloc[-1])**(1/(len(v)-1))-1 if len(v)>1 and v.iloc[-1] else None

def chunk_df(df, size=MAX_TICKERS_PER_CHART):
    if df.empty:
        return []
    return [df.iloc[i:i+size] for i in range(0, len(df), size)]

def auto_ylim(ax, values, pad=0.10):
    """Ajuste automático del eje Y."""
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype="float64")
    else:
        arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax == vmin:
        ymin = vmin - abs(vmin)*pad - 1
        ymax = vmax + abs(vmax)*pad + 1
        ax.set_ylim(ymin, ymax)
        return
    if vmin >= 0:
        ymin = 0
        ymax = vmax * (1 + pad)
    elif vmax <= 0:
        ymax = 0
        ymin = vmin * (1 + pad)
    else:
        m = max(abs(vmin), abs(vmax)) * (1 + pad)
        ymin, ymax = -m, m
    ax.set_ylim(ymin, ymax)

def obtener_datos_financieros(tk, Tc_def):
    try:
        tkr = yf.Ticker(tk)
        info = tkr.info
        bs = tkr.balance_sheet
        fin = tkr.financials
        cf = tkr.cashflow
        
        # Datos básicos
        beta = info.get("beta", 1)
        ke = calc_ke(beta)
        
        debt = safe_first(seek_row(bs, ["Total Debt", "Long Term Debt"])) or info.get("totalDebt", 0)
        cash = safe_first(seek_row(bs, [
            "Cash And Cash Equivalents",
            "Cash And Cash Equivalents At Carrying Value",
            "Cash Cash Equivalents And Short Term Investments",
        ]))
        equity = safe_first(seek_row(bs, ["Common Stock Equity", "Total Stockholder Equity"]))

        interest = safe_first(seek_row(fin, ["Interest Expense"]))
        ebt = safe_first(seek_row(fin, ["Ebt", "EBT"]))
        tax_exp = safe_first(seek_row(fin, ["Income Tax Expense"]))
        ebit = safe_first(seek_row(fin, ["EBIT", "Operating Income",
                                       "Earnings Before Interest and Taxes"]))

        kd = calc_kd(interest, debt)
        tax = tax_exp / ebt if ebt else Tc_def
        mcap = info.get("marketCap", 0)
        wacc = calc_wacc(mcap, debt, ke, kd, tax)

        nopat = ebit * (1 - tax) if ebit is not None else None
        invested = (equity or 0) + ((debt or 0) - (cash or 0))
        roic = nopat / invested if (nopat is not None and invested) else None
        
        # CALCULAR CREACIÓN DE VALOR (WACC vs ROIC) en lugar de EVA
        creacion_valor = (roic - wacc) * 100 if all(v is not None for v in (roic, wacc)) else None

        price = info.get("currentPrice")
        fcf = safe_first(seek_row(cf, ["Free Cash Flow"]))
        shares = info.get("sharesOutstanding")
        pfcf = price / (fcf/shares) if (fcf and shares) else None

        # Cálculo de ratios
        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")
        debt_eq = info.get("debtToEquity")
        lt_debt_eq = info.get("longTermDebtToEquity")
        oper_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")
        
        # Dividendos
        div_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")
        
        # Crecimiento
        revenue_growth = cagr4(fin, "Total Revenue")
        eps_growth = cagr4(fin, "Net Income")
        fcf_growth = cagr4(cf, "Free Cash Flow") or cagr4(cf, "Operating Cash Flow")

        return {
            "Ticker": tk,
            "Nombre": info.get("longName") or info.get("shortName") or info.get("displayName") or tk,
            "País": info.get("country") or info.get("countryCode") or "N/D",
            "Industria": info.get("industry") or info.get("industryKey") or info.get("industryDisp") or "N/D",
            "Sector": info.get("sector", "Unknown"),
            "Precio": price,
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "P/FCF": pfcf,
            "Dividend Yield %": div_yield,
            "Payout Ratio": payout,
            "ROA": roa,
            "ROE": roe,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "Debt/Eq": debt_eq,
            "LtDebt/Eq": lt_debt_eq,
            "Oper Margin": oper_margin,
            "Profit Margin": profit_margin,
            "WACC": wacc,
            "ROIC": roic,
            "Creacion Valor (Wacc vs Roic)": creacion_valor,
            "Revenue Growth": revenue_growth,
            "EPS Growth": eps_growth,
            "FCF Growth": fcf_growth,
            "MarketCap": mcap
        }
    except Exception as e:
        # Manejar específicamente el error de rate limiting
        if "Too Many Requests" in str(e) or "rate" in str(e).lower():
            st.warning(f"Rate limit alcanzado para {tk}. Reintentando después de pausa...")
            time.sleep(2)  # Pausa más larga para rate limiting
            # Reintentar una vez más después de la pausa
            try:
                time.sleep(2)
                return obtener_datos_financieros(tk, Tc_def)
            except:
                st.error(f"Error persistente obteniendo datos para {tk}: {str(e)}")
                return None
        else:
            st.error(f"Error obteniendo datos para {tk}: {str(e)}")
            return None

# =============================================================
# INTERFAZ PRINCIPAL
# =============================================================
def main():
    # Encabezado principal con diseño mejorado
    st.markdown('<h1 class="main-header">📊 Dashboard de Análisis Financiero Avanzado</h1>', unsafe_allow_html=True)
    
    # Información destacada
    st.markdown("""
    <div class="info-text">
        <h3>💡 Supermercado de Inversiones</h3>
        <p>Analizá más de 150 alternativas de inversión con tu cuenta comitente. 
        Nuestra plataforma te ofrece herramientas avanzadas para tomar decisiones informadas.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar con diseño mejorado
    with st.sidebar:
        st.markdown("""
        <div style="background-color: #1976D2; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
            <h2 style="color: white; margin: 0;">⚙️ Configuración</h2>
        </div>
        """, unsafe_allow_html=True)
        
        t_in = st.text_area("**Tickers** (separados por comas)", 
                          "HRL, AAPL, MSFT, ABT, O, XOM, KO, JNJ, CLX, CHD, CB, DDOG")
        max_t = st.slider("**Máximo de tickers**", 1, 50, 12)
        
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #E3F2FD; padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h3 style="color: #0D47A1; margin: 0;">📈 Parámetros WACC</h3>
        </div>
        """, unsafe_allow_html=True)
        
        global Rf, Rm, Tc0
        Rf = st.number_input("**Tasa libre de riesgo (%)**", 0.0, 20.0, 4.35)/100
        Rm = st.number_input("**Retorno esperado del mercado (%)**", 0.0, 30.0, 8.5)/100
        Tc0 = st.number_input("**Tasa impositiva corporativa (%)**", 0.0, 50.0, 21.0)/100

    # Botón de análisis con diseño mejorado
    if st.button("🔍 Analizar Acciones", type="primary"):
        tickers = [t.strip().upper() for t in t_in.split(",") if t.strip()][:max_t]
        
        # Obtener datos
        datos = []
        errs = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Obteniendo datos financieros..."):
            for i, tk in enumerate(tickers):
                try:
                    status_text.text(f"⏳ Procesando {tk} ({i+1}/{len(tickers)})...")
                    data = obtener_datos_financieros(tk, Tc0)
                    if data:
                        datos.append(data)
                    # Aumentar el tiempo de espera a 1 segundo entre solicitudes
                    time.sleep(1)
                except Exception as e:
                    errs.append({"Ticker": tk, "Error": str(e)})
                progress_bar.progress((i + 1) / len(tickers))

        status_text.text("✅ Análisis completado!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        if not datos:
            st.error("No se pudieron obtener datos para los tickers proporcionados")
            if errs:
                st.table(pd.DataFrame(errs))
            return

        df = pd.DataFrame(datos)
        df["SectorRank"] = df["Sector"].map(SECTOR_RANK).fillna(99).astype(int)
        df = df.sort_values(["SectorRank", "Sector", "Ticker"])
        
        # Formatear valores para visualización
        df_disp = df.copy()
        
        # Columnas con 2 decimales
        for col in ["P/E", "P/B", "P/FCF", "Current Ratio", "Quick Ratio", "Debt/Eq", "LtDebt/Eq"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2))
            
        # Porcentajes con 2 decimales
        for col in ["Dividend Yield %", "Payout Ratio", "ROA", "ROE", "Oper Margin", 
                   "Profit Margin", "WACC", "ROIC", "Revenue Growth", "EPS Growth", "FCF Growth"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2, is_percent=True))
            
        # Creación de Valor con 2 decimales y porcentaje
        df_disp["Creacion Valor (Wacc vs Roic)"] = df_disp["Creacion Valor (Wacc vs Roic)"].apply(
            lambda x: format_number(x/100, 2, is_percent=True) if pd.notnull(x) else "N/D"
        )
            
        # Precio y MarketCap con 2 decimales
        df_disp["Precio"] = df_disp["Precio"].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "N/D")
        df_disp["MarketCap"] = df_disp["MarketCap"].apply(lambda x: f"${float(x)/1e9:,.2f}B" if pd.notnull(x) else "N/D")
        
        # Asegurar que las columnas de texto no sean None
        for c in ["Nombre", "País", "Industria"]:
            df_disp[c] = df_disp[c].fillna("N/D").replace({None: "N/D", "": "N/D"})

        # =====================================================
        # SECCIÓN 1: RESUMEN GENERAL
        # =====================================================
        st.markdown('<h2 class="sub-header">📋 Resumen General (agrupado por Sector)</h2>', unsafe_allow_html=True)
        
        # Mostrar tabla
        st.dataframe(
            df_disp[[
                "Ticker", "Nombre", "País", "Industria", "Sector",
                "Precio", "P/E", "P/B", "P/FCF",
                "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                "WACC", "ROIC", "Creacion Valor (Wacc vs Roic)", "MarketCap"
            ]],
            use_container_width=True,
            height=500
        )

        if errs:
            st.markdown('<h3 class="sub-header">🚫 Tickers con error</h3>', unsafe_allow_html=True)
            st.table(pd.DataFrame(errs))

        sectors_ordered = df["Sector"].unique()

        # =====================================================
        # SECCIÓN 2: ANÁLISIS DE VALORACIÓN
        # =====================================================
        st.markdown('<h2 class="sub-header">💰 Análisis de Valoración (por Sector)</h2>', unsafe_allow_html=True)
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec} ({len(sec_df)} empresas)", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 4))
                val = sec_df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                val.plot(kind="bar", ax=ax, rot=45)
                ax.set_ylabel("Ratio")
                auto_ylim(ax, val)
                st.pyplot(fig)
                plt.close()

        # =====================================================
        # SECCIÓN 3: RENTABILIDAD Y EFICIENCIA
        # =====================================================
        st.markdown('<h2 class="sub-header">📈 Rentabilidad y Eficiencia</h2>', unsafe_allow_html=True)
        
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    rr = pd.DataFrame({
                        "ROE": (sec_df["ROE"]*100).values,
                        "ROA": (sec_df["ROA"]*100).values
                    }, index=sec_df["Ticker"])
                    rr.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, rr)
                    st.pyplot(fig)
                    plt.close()

        with tabs[1]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mm = pd.DataFrame({
                        "Oper Margin": (sec_df["Oper Margin"]*100).values,
                        "Profit Margin": (sec_df["Profit Margin"]*100).values
                    }, index=sec_df["Ticker"])
                    mm.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, mm)
                    st.pyplot(fig)
                    plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(12, 6))
            rw = pd.DataFrame({
                "ROIC": (df["ROIC"]*100).values,
                "WACC": (df["WACC"]*100).values
            }, index=df["Ticker"])
            rw.plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            ax.set_title("Creación de Valor: ROIC vs WACC")
            auto_ylim(ax, rw)
            st.pyplot(fig)
            plt.close()

        # =====================================================
        # SECCIÓN 4: ESTRUCTURA DE CAPITAL Y LIQUIDEZ
        # =====================================================
        st.markdown('<h2 class="sub-header">🏦 Estructura de Capital y Liquidez (por sector)</h2>', unsafe_allow_html=True)
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.caption("Apalancamiento")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        lev = chunk[["Ticker", "Debt/Eq", "LtDebt/Eq"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                        lev.plot(kind="bar", stacked=True, ax=ax, rot=45)
                        ax.axhline(1, color="red", linestyle="--")
                        ax.set_ylabel("Ratio")
                        auto_ylim(ax, lev)
                        st.pyplot(fig)
                        plt.close()
                        
                    with c2:
                        st.caption("Liquidez")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        liq = chunk[["Ticker", "Current Ratio", "Quick Ratio"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                        liq.plot(kind="bar", ax=ax, rot=45)
                        ax.axhline(1, color="green", linestyle="--")
                        ax.set_ylabel("Ratio")
                        auto_ylim(ax, liq)
                        st.pyplot(fig)
                        plt.close()

        # =====================================================
        # SECCIÓN 5: CRECIMIENTO
        # =====================================================
        st.markdown('<h2 class="sub-header">🚀 Crecimiento (CAGR 3-4 años, por sector)</h2>', unsafe_allow_html=True)
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    gdf = pd.DataFrame({
                        "Revenue Growth": (chunk["Revenue Growth"]*100).values,
                        "EPS Growth": (chunk["EPS Growth"]*100).values,
                        "FCF Growth": (chunk["FCF Growth"]*100).values
                    }, index=chunk["Ticker"])
                    gdf.plot(kind="bar", ax=ax, rot=45)
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("%")
                    auto_ylim(ax, gdf)
                    st.pyplot(fig)
                    plt.close()

        # =====================================================
        # SECCIÓN 6: ANÁLISIS INDIVIDUAL
        # =====================================================
        st.markdown('<h2 class="sub-header">🔍 Análisis por Empresa</h2>', unsafe_allow_html=True)
        pick = st.selectbox("Selecciona empresa", df_disp["Ticker"].unique())
        det_disp = df_disp[df_disp["Ticker"] == pick].iloc[0]
        det_raw = df[df["Ticker"] == pick].iloc[0]

        st.markdown(f"""
        <div class="ticker-card">
            <h3>{det_raw['Nombre']}</h3>
            <p><strong>Sector:</strong> {det_raw['Sector']}</p>
            <p><strong>País:</strong> {det_raw['País']}</p>
            <p><strong>Industria:</strong> {det_raw['Industria']}</p>
        </div>
        """, unsafe_allow_html=True)

        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precio", det_disp["Precio"])
            st.metric("P/E", det_disp["P/E"])
            st.metric("P/B", det_disp["P/B"])
            st.metric("P/FCF", det_disp["P/FCF"])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with cB:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Market Cap", det_disp["MarketCap"])
            st.metric("ROIC", det_disp["ROIC"])
            st.metric("WACC", det_disp["WACC"])
            
            # Determinar clase CSS según creación de valor
            valor_class = "positive-value" if det_raw.get("Creacion Valor (Wacc vs Roic)", 0) > 0 else "negative-value"
            valor_text = f'<span class="{valor_class}">{det_disp["Creacion Valor (Wacc vs Roic)"]}</span>'
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <label style="font-weight: 600;">Creación Valor</label>
                <div style="font-size: 1.5rem; margin-top: 0.5rem;">{valor_text}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with cC:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("ROE", det_disp["ROE"])
            st.metric("Dividend Yield", det_disp["Dividend Yield %"])
            st.metric("Current Ratio", det_disp["Current Ratio"])
            st.metric("Debt/Eq", det_disp["Debt/Eq"])
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("ROIC vs WACC")
        if pd.notnull(det_raw["ROIC"]) and pd.notnull(det_raw["WACC"]):
            fig, ax = plt.subplots(figsize=(5, 4))
            comp = pd.DataFrame({
                "ROIC": [det_raw["ROIC"]*100],
                "WACC": [det_raw["WACC"]*100]
            }, index=[pick])
            comp.plot(kind="bar", ax=ax, rot=0, legend=False, 
                     color=["green" if det_raw["ROIC"] > det_raw["WACC"] else "red", "gray"])
            ax.set_ylabel("%")
            auto_ylim(ax, comp)
            st.pyplot(fig)
            plt.close()
            
            if det_raw["ROIC"] > det_raw["WACC"]:
                st.success("✅ Crea valor (ROIC > WACC)")
            else:
                st.error("❌ Destruye valor (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para comparar ROIC/WACC")

if __name__ == "__main__":
    main()
