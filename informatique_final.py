import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq
from scipy.signal import argrelextrema

# ============================================================
# Projet : Van der Waals + Construction de Maxwell (Streamlit)
# - Version web : graphique interactif via Plotly
# - Unités utilisées ici : P en bar, V en L/mol, T en K
# ============================================================

# ==========================================
# CONFIGURATION ET DESIGN DE LA PAGE
# ==========================================
st.set_page_config(page_title="Modélisation des Gaz Réels", layout="wide")

st.markdown("""
    <style>
    /* Design académique et épuré */
    .stApp { background-color: #FAFAFA; color: #1E1E1E; }
    h1 { color: #1A237E !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; border-bottom: 2px solid #1A237E; padding-bottom: 10px; font-size: 2.2rem !important; }
    h2, h3 { color: #0D47A1 !important; margin-top: 20px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    /* Style des onglets */
    .stTabs [data-baseweb="tab-list"] { background-color: #E3F2FD; border-radius: 4px; padding: 2px; }
    .stTabs [data-baseweb="tab"] { font-weight: bold; color: #1A237E; font-size: 1.05rem; }
    .stTabs [aria-selected="true"] { background-color: #1A237E !important; color: white !important; border-radius: 4px; }
    
    /* Tableaux explicatifs */
    table { width: 100%; border-collapse: collapse; margin-bottom: 20px; background-color: white; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 0.95rem; }
    th { background-color: #1A237E; color: white; padding: 10px; border: 1px solid #ddd; text-align: left; }
    td { padding: 10px; border: 1px solid #ddd; vertical-align: top; }
    .content-card { background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; text-align: justify; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTEUR PHYSIQUE ET MATHÉMATIQUE
# ==========================================
# Constante des gaz parfaits dans les unités choisies (L·bar·mol⁻¹·K⁻¹)
R = 0.08314

def P_vdw(V, T, a, b):
    """Pression de Van der Waals."""
    return (R * T) / (V - b) - a / (V**2)

def P_ideal(V, T):
    """Pression d'un gaz parfait (référence de comparaison)."""
    return (R * T) / V

def primitive_VdW(V, T, a, b):
    """Primitive analytique de P_vdw(V) par rapport à V pour le calcul des aires."""
    return R * T * np.log(np.abs(V - b)) + a / V

def difference_aires(P_test, T, a, b):
    """Fonction objectif de Maxwell.
    On impose : aire_courbe(VL→VG) - aire_rectangle = 0.
    """
    coeffs = [P_test, -(P_test * b + R * T), a, -a * b]
    roots = np.roots(coeffs)
    r_reelles = np.sort(roots[np.isreal(roots)].real)
    
    if len(r_reelles) < 3: 
        return 1e9
        
    aire_courbe = primitive_VdW(r_reelles[2], T, a, b) - primitive_VdW(r_reelles[0], T, a, b)
    aire_rect = P_test * (r_reelles[2] - r_reelles[0])
    return aire_courbe - aire_rect

def trouver_plateau(T, a, b):
    """Trouve la pression de saturation Psat via la méthode de Brent."""
    v_scan = np.linspace(b + 0.001, max(1.5, b*100), 5000)
    p_scan = P_vdw(v_scan, T, a, b)
    
    idx_max = argrelextrema(p_scan, np.greater)[0]
    idx_min = argrelextrema(p_scan, np.less)[0]
    
    if len(idx_max) == 0 or len(idx_min) == 0:
        return None 
        
    P_high = p_scan[idx_max[0]]
    P_low = max(p_scan[idx_min[0]], 0.01)
    
    try:
        Psat = brentq(difference_aires, P_low + 0.001, P_high - 0.001, args=(T, a, b))
        coeffs = [Psat, -(Psat * b + R * T), a, -a * b]
        roots_fin = np.sort(np.roots(coeffs)[np.isreal(np.roots(coeffs))].real)
        return Psat, roots_fin[0], roots_fin[1], roots_fin[2]
    except ValueError:
        return None

def coordonnees_critiques(a, b):
    """Calcule les constantes critiques théoriques."""
    Tc = (8 * a) / (27 * R * b)
    Vc = 3 * b
    Pc = a / (27 * b**2)
    return Tc, Vc, Pc

# ==========================================
# 2. MENU LATÉRAL
# ==========================================
with st.sidebar:
    st.title("Paramètres de l'Étude")
    mode = st.radio("Choix du Mode", ["Isotherme Unique (Analyse d'Aires)", "Diagramme de Phase (Réseau)"])
    st.markdown("---")
    st.subheader("Substance étudiée")
    
    choix_gaz = st.selectbox("Fluide pré-enregistré", ["CO2", "H2O", "O2", "N2", "He", "Personnalisé"])
    if choix_gaz == "CO2": a_def, b_def = 3.640, 0.0427
    elif choix_gaz == "H2O": a_def, b_def = 5.530, 0.0305
    elif choix_gaz == "O2": a_def, b_def = 1.382, 0.0318
    elif choix_gaz == "N2": a_def, b_def = 1.370, 0.0387
    elif choix_gaz == "He": a_def, b_def = 0.0346, 0.0238
    else: a_def, b_def = 3.640, 0.0427
    
    if choix_gaz == "Personnalisé":
        a = st.number_input("Paramètre 'a' (L².bar/mol²)", value=a_def, format="%.3f")
        b = st.number_input("Paramètre 'b' (L/mol)", value=b_def, format="%.4f")
    else:
        a, b = a_def, b_def

    Tc_crit, Vc_crit, Pc_crit = coordonnees_critiques(a, b)
    st.markdown("---")
    st.subheader("Point Critique Théorique")
    st.info(f"**Tc** : {Tc_crit:.2f} K\n\n**Pc** : {Pc_crit:.2f} bar\n\n**Vc** : {Vc_crit:.4f} L/mol")

# ==========================================
# 3. INTERFACE PRINCIPALE
# ==========================================
st.markdown("<h1>Modélisation d'un Gaz Réel : Équation de Van der Waals</h1>", unsafe_allow_html=True)

tab_sim, tab_theo, tab_methode, tab_code = st.tabs([
    "Simulateur Interactif", 
    "Théorie & Interprétation", 
    "Analyse du Programme", 
    "Code Source"
])

# -------------------------------------------------------------------------------------
# ONGLET 1 : SIMULATEUR
# -------------------------------------------------------------------------------------
with tab_sim:
    fig = go.Figure()
    V_plot = np.linspace(b * 1.05, Vc_crit * 6, 1500)
    hover_style = dict(bgcolor="white", bordercolor="black", font=dict(color="black"))

    if mode == "Isotherme Unique (Analyse d'Aires)":
        st.subheader("Étude d'une Isotherme de Van der Waals et Construction de Maxwell")
        T = st.slider("Température de l'isotherme (K)", 100.0, float(int(Tc_crit*1.5)), float(int(Tc_crit*0.85)))
        resultat = trouver_plateau(T, a, b)
        
        fig.add_trace(go.Scatter(x=V_plot, y=P_ideal(V_plot, T), line=dict(color='black', dash='dash', width=1.5), opacity=0.5, name="Gaz Parfait"))

        if resultat:
            P_sat, V_liq, V_mid, V_gaz = resultat
            
            fig.add_trace(go.Scatter(x=V_plot, y=P_vdw(V_plot, T, a, b), line=dict(color='blue', width=2), opacity=0.5, name="Van der Waals",
                                     customdata=[T]*len(V_plot), hoverlabel=hover_style, hovertemplate="<b>VdW</b><br>T = %{customdata:.1f} K<br>V = %{x:.4f}<br>P = %{y:.2f} bar<extra></extra>"))
            
            fig.add_trace(go.Scatter(x=[V_liq, V_gaz], y=[P_sat, P_sat], line=dict(color='red', width=3), name=f"Plateau de Saturation"))
            
            v_bosse = np.linspace(V_liq, V_mid, 100)
            x_bosse = np.concatenate([v_bosse, v_bosse[::-1]])
            y_bosse = np.concatenate([P_vdw(v_bosse, T, a, b), np.full_like(v_bosse, P_sat)])
            fig.add_trace(go.Scatter(x=x_bosse, y=y_bosse, fill='toself', fillcolor='rgba(255, 165, 0, 0.4)', line=dict(width=0), name="Aire Inférieure", hoverinfo='skip'))
            
            v_creux = np.linspace(V_mid, V_gaz, 100)
            x_creux = np.concatenate([v_creux, v_creux[::-1]])
            y_creux = np.concatenate([np.full_like(v_creux, P_sat), P_vdw(v_creux, T, a, b)[::-1]])
            fig.add_trace(go.Scatter(x=x_creux, y=y_creux, fill='toself', fillcolor='rgba(0, 128, 0, 0.4)', line=dict(width=0), name="Aire Supérieure", hoverinfo='skip'))
            
            fig.add_trace(go.Scatter(x=[V_liq], y=[P_sat], mode='markers', marker=dict(color='blue', size=10, line=dict(color='black', width=1)), name="Liquide Saturé",
                                     hoverlabel=hover_style, hovertemplate="<b>Liquide Saturé</b><br>V = %{x:.4f} L/mol<br>P = %{y:.2f} bar<extra></extra>"))
            fig.add_trace(go.Scatter(x=[V_gaz], y=[P_sat], mode='markers', marker=dict(color='red', size=10, line=dict(color='black', width=1)), name="Vapeur Saturée",
                                     hoverlabel=hover_style, hovertemplate="<b>Vapeur Saturée</b><br>V = %{x:.4f} L/mol<br>P = %{y:.2f} bar<extra></extra>"))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Pression de Saturation", f"{P_sat:.2f} bar")
            col2.metric("Volume Liquide saturé", f"{V_liq:.4f} L/mol")
            col3.metric("Volume Vapeur saturée", f"{V_gaz:.4f} L/mol")
            
            fig.update_layout(yaxis=dict(range=[0, P_sat * 2.5]))
        else:
            fig.add_trace(go.Scatter(x=V_plot, y=P_vdw(V_plot, T, a, b), line=dict(color='blue', width=2), opacity=0.5, name="Van der Waals"))
            st.warning("État Supercritique : Pas de transition de phase à cette température.")
            fig.update_layout(yaxis=dict(range=[0, Pc_crit * 2]))

    elif mode == "Diagramme de Phase (Réseau)":
        st.subheader("Diagramme de Phase Complet (Courbe Binodale)")
        
        temps_cloche = np.linspace(Tc_crit*0.65, Tc_crit * 0.99, 12)
        l_vl, l_vg, l_psat = [], [], []
        
        for T_i in temps_cloche:
            res = trouver_plateau(T_i, a, b)
            if res:
                ps, vl, vm, vg = res
                l_vl.append(vl); l_vg.append(vg); l_psat.append(ps)
                
                fig.add_trace(go.Scatter(x=V_plot, y=P_vdw(V_plot, T_i, a, b), mode='lines', line=dict(color='steelblue', width=1.2), opacity=0.5, showlegend=False, 
                                         customdata=[T_i]*len(V_plot), hoverlabel=hover_style, hovertemplate="<b>Isotherme</b><br>T = %{customdata:.1f} K<br>V = %{x:.4f}<br>P = %{y:.2f} bar<extra></extra>"))
                fig.add_trace(go.Scatter(x=[vl, vg], y=[ps, ps], mode='lines', line=dict(color='crimson', width=2), showlegend=False, hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=l_vl + [Vc_crit] + l_vg[::-1], y=l_psat + [Pc_crit] + l_psat[::-1], line=dict(color='black', width=2.5, dash='dash'), name="Courbe Binodale"))
        
        fig.add_trace(go.Scatter(x=l_vl, y=l_psat, mode='markers', marker=dict(color='blue', size=8, line=dict(color='white', width=0.5)), name="Liquide saturé",
                                 hoverlabel=hover_style, customdata=temps_cloche, hovertemplate="<b>Liquide saturé</b><br>T = %{customdata:.1f} K<br>V = %{x:.4f}<br>P = %{y:.2f} bar<extra></extra>"))
        fig.add_trace(go.Scatter(x=l_vg, y=l_psat, mode='markers', marker=dict(color='red', size=8, line=dict(color='white', width=0.5)), name="Vapeur saturée",
                                 hoverlabel=hover_style, customdata=temps_cloche, hovertemplate="<b>Vapeur saturée</b><br>T = %{customdata:.1f} K<br>V = %{x:.4f}<br>P = %{y:.2f} bar<extra></extra>"))
        
        fig.add_trace(go.Scatter(x=[Vc_crit], y=[Pc_crit], mode='markers', marker=dict(color='black', size=12), name='Point Critique', 
                                 hoverlabel=hover_style, hovertemplate=f"<b>Point Critique</b><br>Tc = {Tc_crit:.1f} K<br>Vc = {Vc_crit:.4f}<br>Pc = {Pc_crit:.2f} bar<extra></extra>"))
        
        fig.update_layout(yaxis=dict(range=[0, Pc_crit * 1.8]))

    fig.update_layout(height=700, paper_bgcolor='#FAFAFA', plot_bgcolor='white', xaxis=dict(title="Volume Molaire (L/mol)", range=[0, Vc_crit*5], gridcolor='#E0E0E0'), yaxis=dict(title="Pression (bar)", gridcolor='#E0E0E0'), legend=dict(x=0.75, y=0.95, bordercolor="black", borderwidth=1), hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------------------
# ONGLET 2 : THÉORIE ET INTERPRÉTATION 
# -------------------------------------------------------------------------------------
with tab_theo:
    st.markdown("""
    <div class="content-card">
    
    ## Introduction
    L'étude du comportement des gaz repose sur l'analyse des relations entre la pression P, le volume V et la température T. Ces grandeurs thermodynamiques permettent de décrire l'état d'un système et d'interpréter son évolution lors de transformations physiques. Afin de modéliser ces relations, plusieurs équations d'état ont été développées, dont le modèle du gaz parfait et l'équation de Van der Waals. Le premier constitue une approximation idéale, tandis que le second permet une description plus réaliste du comportement des gaz réels.

    ## 1. Le modèle du gaz parfait
    Le modèle du gaz parfait est une approximation simple qui repose sur l'hypothèse que les molécules d'un gaz sont des particules ponctuelles sans volume propre et qu'elles n'interagissent pas entre elles. Dans ce cadre, les seules forces qui entrent en jeu sont celles résultant des collisions élastiques entre molécules et avec les parois du récipient.
    L'équation d'état qui décrit ce modèle est :
    $$PV=nRT$$
    À température constante (isotherme), on obtient :
    $$P(V)=\\frac{nRT}{V}$$
    Cette relation implique que, à température constante, le produit PV est une constante et que la pression diminue de façon hyperbolique lorsque le volume augmente. Cette loi rend bien compte du comportement des gaz dilués à faible pression et haute température, mais elle présente des limites importantes lorsque les conditions expérimentales s'éloignent de ce domaine.

    ## 2. Limites du modèle du gaz parfait
    Le modèle du gaz parfait ne tient pas compte de deux aspects physiques essentiels des gaz réels :
    * les molécules occupent un volume non négligeable, surtout lorsque la densité augmente ;
    * il existe des interactions attractives entre molécules, en particulier à basse température ou forte pression.
    Dans ces conditions, le comportement des gaz réels s'écarte de celui des gaz parfaits : les courbes P(V) ne sont plus de simples hyperboles, et des phénomènes tels que la liquéfaction ou l'existence d'un point critique apparaissent expérimentalement.

    ## 3. L'équation d'état de Van der Waals
    Afin de corriger ces limites, Johannes Diderik van der Waals a proposé en 1873 une équation d'état plus réaliste qui tient compte des effets microscopiques non inclus dans le modèle parfait. Contrairement au gaz parfait, cette équation considère que les molécules ont un volume propre et qu'il existe des forces d'attraction entre elles.
    La forme de l'équation de Van der Waals est :
    $$P=\\frac{nRT}{V-nb}-\\frac{an^2}{V^2}$$
    * **Le paramètre b** représente le volume propre des molécules. Contrairement au modèle du gaz parfait, les particules ne sont donc plus considérées comme ponctuelles. Le volume réellement accessible au mouvement des molécules n'est plus V, mais V-nb. Ainsi, lorsque le volume total tend vers nb, le terme V-nb devient très faible, ce qui entraîne une augmentation brutale de la pression.
    * **Le paramètre a**, quant à lui, traduit l'existence d'interactions attractives entre les molécules (forces de cohésion). Ces forces réduisent la pression effective exercée sur les parois du récipient, car une partie de l'énergie des particules est mobilisée pour compenser ces attractions. Par conséquent, plus la valeur de a est grande, plus les interactions attractives sont importantes et plus la pression calculée diminue.

    ## Construction de Maxwell
    Lorsque la température est inférieure à la température critique (T < Tc), l'équation de Van der Waals prédit une isotherme présentant une portion en forme de « S ». Cette forme apparaît car, dans une zone intermédiaire, la pente devient positive : $\\frac{dP}{dV}>0$.
    Cela signifie que la pression augmente lorsque le volume augmente, ce qui correspond à un état mécaniquement instable. En effet, pour un système stable, la pression doit diminuer lorsque le volume augmente. Les points qui délimitent cette zone sont obtenus en résolvant $\\frac{dP}{dV}=0$. Ils correspondent au maximum et au minimum local de l'isotherme.
    Cependant, expérimentalement, lors de la transition liquide-gaz, la transformation se fait à pression constante. Pour corriger la zone instable prédite par le modèle, on applique la construction de Maxwell, qui impose la condition :
    $$\\int_{V_L}^{V_G} (P(V) - P_{sat}) dV = 0$$
    Cette condition signifie que les aires situées au-dessus et en dessous du palier horizontal sont égales. Elle garantit l'équilibre thermodynamique entre les phases liquide et vapeur. Ainsi, la dérivée permet d'identifier la zone instable, tandis que l'intégrale permet de déterminer la pression de saturation correspondant à la transition réelle observée.

    ## 4. Température critique, pression critique et volume critique
    Dans le modèle de Van der Waals, le point critique correspond à l'état pour lequel la distinction entre phase liquide et phase vapeur disparaît. Sur le diagramme P(V) ce point se caractérise par un point d'inflexion de l'isotherme critique. Mathématiquement, cela signifie que la première et la seconde dérivées de la pression par rapport au volume s'annulent simultanément :
    $$\\frac{dP}{dV}=0 \\quad \\text{et} \\quad \\frac{d^2P}{dV^2}=0$$
    En appliquant ces conditions à l'équation de Van der Waals, on obtient les expressions des grandeurs critiques :
    $$T_c=\\frac{8a}{27Rb} \\quad ; \\quad P_c=\\frac{a}{27b^2} \\quad ; \\quad V_c=3b$$

    ## 5. Interprétation générale des diagrammes P=f(V)
    Les diagrammes obtenus représentent les isothermes d'un gaz réel décrites par l'équation de Van der Waals, avec correction de la zone instable par la construction de Maxwell. Chaque couleur correspond à une température différente et permet d'identifier un régime thermodynamique particulier.
    
    * **Courbes bleues : températures inférieures à Tc**
    Les courbes bleues correspondent aux températures inférieures à la température critique. Pour ces températures, on observe un palier horizontal au centre de chaque courbe. Ce palier correspond à la transition liquide-gaz. Lorsque le système est comprimé ou détendu dans cette zone : le volume change fortement, mais la pression reste constante. Cela signifie que le fluide n'est pas entièrement liquide ni entièrement vapeur : les deux phases coexistent. Plus la température est basse, plus le palier est long. Cela s'explique par le fait que la différence entre le volume molaire du liquide VL et celui de la vapeur VG est plus importante.
    
    * **Courbe noire : isotherme critique (T=Tc)**
    La courbe noire correspond à l'isotherme tracée pour la température critique T=Tc. Lorsque la température augmente progressivement à partir d'une valeur inférieure à Tc, on observe que le palier horizontal devient de plus en plus court. À la température critique, ces deux volumes deviennent égaux ($V_L=V_G=V_c$). Autrement dit, la phase liquide et la phase vapeur deviennent indiscernables. Graphiquement, cela se traduit par la disparition complète du palier horizontal. La courbe ne présente plus de segment à pression constante, mais un point particulier appelé point d'inflexion. Au-delà de Tc, il n'y a plus de transition liquide-gaz possible : le fluide entre dans le domaine supercritique.
    
    * **Courbes rouges et orangées : isothermes pour T>Tc**
    Les courbes rouges et orangées correspondent aux températures supérieures à la température critique. On remarque qu'il n'y a plus de palier horizontal. La pression diminue continûment lorsque le volume augmente. Cela signifie qu'aucune transition liquide-gaz n'est possible dans ce domaine. Le fluide ne peut plus se séparer en deux phases distinctes. Il est dans un état supercritique. Dans cet état : il n'existe plus de frontière nette entre liquide et gaz, le fluide possède des propriétés intermédiaires entre les deux.
    
    * **Partie gauche des courbes (faibles volumes)**
    À faible volume, toutes les courbes (quelle que soit la couleur) montent très fortement. Cela signifie que la pression augmente rapidement lorsque le volume diminue. Ce comportement est lié au paramètre b, qui représente le volume propre des molécules. Lorsque le volume total se rapproche de b, il devient très difficile de comprimer davantage le fluide, ce qui provoque une augmentation importante de la pression.
    
    * **Partie droite des courbes (grands volumes)**
    À grand volume, toutes les courbes deviennent plus proches les unes des autres. Dans cette région, les molécules sont très éloignées, les interactions attractives deviennent négligeables. Le comportement du fluide se rapproche alors de celui d'un gaz parfait, où la pression diminue progressivement avec le volume.

    ## Conclusion
    Ce projet a permis d'étudier le comportement des gaz réels à l'aide de l'équation de Van der Waals et de sa mise en œuvre numérique. Les diagrammes P=f(V) obtenus mettent en évidence la transition liquide-gaz pour T<Tc, la disparition de cette transition au point critique et le comportement supercritique pour T>Tc. L'application de la construction de Maxwell a permis de corriger la zone instable prédite par le modèle théorique et d'obtenir des courbes cohérentes avec la réalité physique. Ce travail illustre ainsi le lien entre théorie thermodynamique, calcul numérique et interprétation graphique.
    
                
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------
# ONGLET 3 : ANALYSE DU PROGRAMME (STRATÉGIE + LIGNES CLÉS)
# -------------------------------------------------------------------------------------
with tab_methode:
    st.markdown("## 5. Description détaillée du programme")
    
    st.markdown("""
    <div class="content-card">
    
    ### Évolution technique : De Matplotlib à Plotly
    Pourquoi avoir changé de bibliothèques par rapport à une version locale classique ? 
    Dans un premier temps, le tracé des courbes s'effectuait via <code>matplotlib</code>. Cependant, pour permettre un déploiement web fluide et accessible à tous via <code>streamlit</code>, nous avons migré vers la bibliothèque <code>plotly.graph_objects</code>. 
    Ce changement technique majeur permet d'obtenir des graphiques vectoriels natifs pour le web. Cela offre une interactivité indispensable pour l'utilisateur final : zoom fluide sans rechargement, affichage dynamique des coordonnées (T, V, P) au simple survol de la souris, et remplissage des aires de Maxwell grandement simplifié.
    
    <br>Le tableau ci-dessous présente l'explication ligne par ligne des commandes majeures utilisées pour construire le solveur mathématique et l'interface web interactive.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table>
        <tr>
            <th>Ligne(s)</th>
            <th>Commande utilisée</th>
            <th>Rôle</th>
        </tr>
        <tr>
            <td>1</td>
            <td><code>import streamlit as st</code></td>
            <td>Génère l'interface web interactive (création de la page, barre latérale, onglets, curseurs).</td>
        </tr>
        <tr>
            <td>2</td>
            <td><code>import numpy as np</code></td>
            <td>Permet d'utiliser NumPy pour effectuer des calculs numériques efficaces (créer le tableau des volumes et résoudre l'équation polynomiale).</td>
        </tr>
        <tr>
            <td>3</td>
            <td><code>import plotly.graph_objects as go</code></td>
            <td>Importe l'outil de visualisation pour générer des graphiques vectoriels interactifs (zoom, et affichage des coordonnées T, V, P au survol de la souris).</td>
        </tr>
        <tr>
            <td>4-5</td>
            <td><code>from scipy... import brentq, argrelextrema</code></td>
            <td>Importe les méthodes numériques pour trouver la racine exacte (annulation des aires) et détecter les extrema locaux (bosses et creux de la zone instable).</td>
        </tr>
        <tr>
            <td>16</td>
            <td><code>st.set_page_config(...)</code></td>
            <td>Configure les paramètres globaux de la page web (titre dans le navigateur et affichage en pleine largeur).</td>
        </tr>
        <tr>
            <td>31</td>
            <td><code>def P_vdw(V, T, a, b):</code></td>
            <td>Définition de la fonction implémentant l'équation de Van der Waals. Isole le calcul physique pour l'appeler facilement dans les boucles de tracés.</td>
        </tr>
        <tr>
            <td>44</td>
            <td><code>def difference_aires(P_test, ...)</code></td>
            <td>Définit la fonction calculant la différence entre l'aire sous l'isotherme et l'aire du rectangle isobare. L'égalité de ces deux aires est la condition de Maxwell.</td>
        </tr>
        <tr>
            <td>49</td>
            <td><code>np.roots(coeffs)</code></td>
            <td>Résout l'équation polynomiale de degré 3 pour trouver les trois volumes d'intersection exacts entre la courbe de Van der Waals et le palier de pression testé.</td>
        </tr>
        <tr>
            <td>58</td>
            <td><code>def trouver_plateau(T, a, b):</code></td>
            <td>Fonction dédiée à la détermination du palier de Maxwell. Elle détecte d'abord l'instabilité, puis cherche la pression de saturation.</td>
        </tr>
        <tr>
            <td>66</td>
            <td><code>argrelextrema(p_scan, ...)</code></td>
            <td>Identifie les extrema locaux de la courbe pour vérifier si l'isotherme possède une zone instable (ce qui arrive seulement si T < Tc).</td>
        </tr>
        <tr>
            <td>75</td>
            <td><code>Psat = brentq(...)</code></td>
            <td>Utilise la méthode d'optimisation de Brent pour trouver la pression exacte qui annule la fonction <code>difference_aires</code>. C'est la pression de saturation.</td>
        </tr>
        <tr>
            <td>89</td>
            <td><code>with st.sidebar:</code></td>
            <td>Crée un panneau latéral à gauche de l'écran pour regrouper proprement les paramètres interactifs de l'utilisateur.</td>
        </tr>
        <tr>
            <td>94</td>
            <td><code>st.selectbox(...)</code></td>
            <td>Affiche un menu déroulant permettant à l'utilisateur de sélectionner un gaz pré-enregistré pour adapter automatiquement <i>a</i> et <i>b</i>.</td>
        </tr>
        <tr>
            <td>116</td>
            <td><code>st.tabs([...])</code></td>
            <td>Sépare l'application en plusieurs onglets (Simulateur, Théorie, Analyse, Code) pour une navigation plus claire et fluide.</td>
        </tr>
        <tr>
            <td>126</td>
            <td><code>fig = go.Figure()</code></td>
            <td>Initialise la fenêtre graphique interactive (équivalent moderne de <code>plt.figure()</code> pour le web).</td>
        </tr>
        <tr>
            <td>131</td>
            <td><code>T = st.slider(...)</code></td>
            <td>Crée un curseur interactif. Chaque mouvement relance les calculs et met à jour le graphique et les valeurs en temps réel.</td>
        </tr>
        <tr>
            <td>139-143</td>
            <td><code>fig.add_trace(go.Scatter(...))</code></td>
            <td>Ajoute une courbe spécifique sur le graphique (trace l'isotherme de Van der Waals, puis trace le palier rouge de saturation).</td>
        </tr>
        <tr>
            <td>147-151</td>
            <td><code>fill='toself'</code></td>
            <td>Propriété graphique permettant de remplir et colorer précisément l'aire sous la courbe (coloration des aires de Maxwell en orange et vert).</td>
        </tr>
        <tr>
            <td>157-160</td>
            <td><code>st.columns(3)</code> / <code>col.metric(...)</code></td>
            <td>Divise l'espace en 3 colonnes pour afficher les grandeurs clés (Pression, V_liq, V_gaz) sous forme de compteurs dynamiques esthétiques.</td>
        </tr>
        <tr>
            <td>183</td>
            <td><code>st.plotly_chart(fig)</code></td>
            <td>Affiche le graphique Plotly finalisé au centre de l'interface web.</td>
        </tr>
        <tr>
            <td>339-340</td>
            <td><code>with open(__file__...) : st.code()</code></td>
            <td>Permet au programme de s'auto-lire ! Il affiche dynamiquement son propre code source .</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------
# ONGLET 4 : CODE SOURCE
# -------------------------------------------------------------------------------------
with tab_code:
    st.markdown("### Code Source de l'Application")
    st.write("Ce script regroupe toute la logique physique de Maxwell et la construction de l'interface visuelle interactive.")
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language='python')
