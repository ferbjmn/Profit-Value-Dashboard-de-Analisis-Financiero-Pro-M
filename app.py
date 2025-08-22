# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO - INTERFAZ MEJORADA
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time

# Configuración global
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 600;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .positive-value {
        color: #2E7D32;
        font-weight: 600;
    }
    .negative-value {
        color: #C62828;
        font-weight: 600;
    }
    .sector-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        background-color: #E3F2FD;
        color: #1565C0;
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
    if pd.isna(x) or x is None:
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
    if fin is None or metric not in fin.index:
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
    # Encabezado principal
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">📊 Dashboard de Análisis Financiero Avanzado</h1>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align: right; padding-top: 1rem;"><small>Powered by Yahoo Finance</small></div>', unsafe_allow_html=True)
    
    st.markdown("Analiza y compara múltiples empresas utilizando métricas financieras clave para la toma de decisiones de inversión.")

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='background-color: #1E88E5; padding: 1rem; border-radius: 0.5rem; color: white; margin-bottom: 1.5rem;'>
            <h3 style='color: white; margin: 0;'>⚙️ Configuración</h3>
        </div>
        """, unsafe_allow_html=True)
        
        t_in = st.text_area("**Tickers** (separados por comas)", 
                          "AAPL, MSFT, GOOGL, AMZN, TSLA")
        max_t = st.slider("**Máximo de tickers**", 1, 50, 12)
        
        st.markdown("---")
        st.markdown("**📈 Parámetros WACC**")
        global Rf, Rm, Tc0
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35)/100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5)/100
        Tc0 = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0)/100
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.8rem; color: #666;'>
            Desarrollado con Streamlit<br>Datos proporcionados por Yahoo Finance
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔍 Analizar Acciones", type="primary", use_container_width=True):
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
        st.markdown('<h2 class="sub-header">📋 Resumen General</h2>', unsafe_allow_html=True)
        
        # Mostrar estadísticas generales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df)}</h3>
                <p>Empresas analizadas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sectors_count = df['Sector'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{sectors_count}</h3>
                <p>Sectores representados</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            avg_pe = df['P/E'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{format_number(avg_pe, 1) if not pd.isna(avg_pe) else 'N/D'}</h3>
                <p>P/E Promedio</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            avg_roe = df['ROE'].mean() * 100 if not df['ROE'].isna().all() else None
            roe_class = "positive-value" if avg_roe and avg_roe > 10 else "negative-value" if avg_roe else ""
            roe_display = f"{avg_roe:.1f}%" if avg_roe else "N/D"
            st.markdown(f"""
            <div class="metric-card">
                <h3 class="{roe_class}">{roe_display}</h3>
                <p>ROE Promedio</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Mostrar tabla
        st.dataframe(
            df_disp[[
                "Ticker", "Nombre", "Sector", "Precio", "P/E", "P/B", "P/FCF",
                "Dividend Yield %", "ROE", "ROIC", "Creacion Valor (Wacc vs Roic)", "MarketCap"
            ]],
            use_container_width=True,
            height=400
        )

        if errs:
            st.error("Algunos tickers no pudieron ser procesados:")
            st.table(pd.DataFrame(errs))

        sectors_ordered = df["Sector"].unique()

        # =====================================================
        # SECCIÓN 2: ANÁLISIS DE VALORACIÓN
        # =====================================================
        st.markdown('<h2 class="sub-header">💰 Análisis de Valoración</h2>', unsafe_allow_html=True)
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"{sec} ({len(sec_df)} empresas)", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 5))
                val = sec_df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                val.plot(kind="bar", ax=ax, rot=45, color=['#1E88E5', '#64B5F6', '#90CAF9'])
                ax.set_ylabel("Ratio")
                ax.set_title(f"Múltiplos de Valoración - Sector {sec}")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
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
                    rr.plot(kind="bar", ax=ax, rot=45, color=['#2E7D32', '#66BB6A'])
                    ax.set_ylabel("%")
                    ax.set_title(f"ROE vs ROA - Sector {sec}")
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
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
                    mm.plot(kind="bar", ax=ax, rot=45, color=['#FF9800', '#FFB74D'])
                    ax.set_ylabel("%")
                    ax.set_title(f"Márgenes Operativos - Sector {sec}")
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    auto_ylim(ax, mm)
                    st.pyplot(fig)
                    plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(12, 6))
            rw = pd.DataFrame({
                "ROIC": (df["ROIC"]*100).values,
                "WACC": (df["WACC"]*100).values
            }, index=df["Ticker"])
            rw.plot(kind="bar", ax=ax, rot=45, color=['#2E7D32', '#C62828'])
            ax.set_ylabel("%")
            ax.set_title("Creación de Valor: ROIC vs WACC")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            auto_ylim(ax, rw)
            st.pyplot(fig)
            plt.close()

        # =====================================================
        # SECCIÓN 4: ESTRUCTURA DE CAPITAL Y LIQUIDEZ
        # =====================================================
        st.markdown('<h2 class="sub-header">🏦 Estructura de Capital y Liquidez</h2>', unsafe_allow_html=True)
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.markdown("**Apalancamiento**")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        lev = chunk[["Ticker", "Debt/Eq", "LtDebt/Eq"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                        lev.plot(kind="bar", stacked=True, ax=ax, rot=45, color=['#5C6BC0', '#9FA8DA'])
                        ax.axhline(1, color="red", linestyle="--", alpha=0.7)
                        ax.set_ylabel("Ratio")
                        ax.set_title("Deuda/Patrimonio")
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        auto_ylim(ax, lev)
                        st.pyplot(fig)
                        plt.close()
                        
                    with c2:
                        st.markdown("**Liquidez**")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        liq = chunk[["Ticker", "Current Ratio", "Quick Ratio"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                        liq.plot(kind="bar", ax=ax, rot=45, color=['#26A69A', '#80CBC4'])
                        ax.axhline(1, color="green", linestyle="--", alpha=0.7)
                        ax.set_ylabel("Ratio")
                        ax.set_title("Ratios de Liquidez")
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        auto_ylim(ax, liq)
                        st.pyplot(fig)
                        plt.close()

        # =====================================================
        # SECCIÓN 5: CRECIMIENTO
        # =====================================================
        st.markdown('<h2 class="sub-header">🚀 Crecimiento (CAGR 3-4 años)</h2>', unsafe_allow_html=True)
        
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
                    gdf.plot(kind="bar", ax=ax, rot=45, color=['#AB47BC', '#7E57C2', '#5C6BC0'])
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("%")
                    ax.set_title(f"Tasas de Crecimiento Anual - Sector {sec}")
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
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
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border: 1px solid #E0E0E0;'>
            <h3>{det_raw['Nombre']}</h3>
            <p><strong>Sector:</strong> <span class="sector-badge">{det_raw['Sector']}</span></p>
            <p><strong>País:</strong> {det_raw['País']} | <strong>Industria:</strong> {det_raw['Industria']}</p>
        </div>
        """, unsafe_allow_html=True)

        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown("**💵 Valoración**")
            st.metric("Precio", det_disp["Precio"])
            st.metric("P/E", det_disp["P/E"])
            st.metric("P/B", det_disp["P/B"])
            st.metric("P/FCF", det_disp["P/FCF"])
            
        with cB:
            st.markdown("**📊 Rentabilidad**")
            st.metric("Market Cap", det_disp["MarketCap"])
            st.metric("ROIC", det_disp["ROIC"])
            st.metric("WACC", det_disp["WACC"])
            
            # Determinar clase CSS para creación de valor
            valor_class = "positive-value" if det_raw.get("Creacion Valor (Wacc vs Roic)", 0) > 0 else "negative-value"
            st.metric("Creación Valor", det_disp["Creacion Valor (Wacc vs Roic)"], delta=None)
            
        with cC:
            st.markdown("**📈 Fundamentales**")
            st.metric("ROE", det_disp["ROE"])
            st.metric("Dividend Yield", det_disp["Dividend Yield %"])
            st.metric("Current Ratio", det_disp["Current Ratio"])
            st.metric("Debt/Eq", det_disp["Debt/Eq"])

        st.subheader("ROIC vs WACC")
        if pd.notnull(det_raw["ROIC"]) and pd.notnull(det_raw["WACC"]):
            fig, ax = plt.subplots(figsize=(6, 5))
            comp = pd.DataFrame({
                "ROIC": [det_raw["ROIC"]*100],
                "WACC": [det_raw["WACC"]*100]
            }, index=[pick])
            colors = ["green" if det_raw["ROIC"] > det_raw["WACC"] else "red", "gray"]
            comp.plot(kind="bar", ax=ax, rot=0, legend=False, color=colors)
            ax.set_ylabel("%")
            ax.set_title("ROIC vs WACC - Creación de Valor")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            auto_ylim(ax, comp)
            st.pyplot(fig)
            plt.close()
            
            if det_raw["ROIC"] > det_raw["WACC"]:
                st.success("✅ **Crea valor** (ROIC > WACC)")
            else:
                st.error("❌ **Destruye valor** (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para comparar ROIC/WACC")

    else:
        # Estado inicial antes de hacer clic en el botón
        st.info("👈 Ingresa los tickers en el panel lateral y haz clic en 'Analizar Acciones' para comenzar.")

if __name__ == "__main__":
    main()
