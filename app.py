import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import pandas as pd
import joblib
import math
import requests

MODEL_PATH = "model/model.joblib"

st.set_page_config(page_title="Heart Disease Risk", layout="centered")


# ---------------------------
# Load model artifact
# ---------------------------
@st.cache_resource
def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], artifact["selected_features"]


# ---------------------------
# Risk scale helper
# ---------------------------
def risk_bucket(p: float):
    """
    Map probability p in [0,1] to:
      - interval label like '0.20 - 0.30'
      - text like 'very safe' ... 'extremely high risk'
    """
    p = max(0.0, min(1.0, float(p)))
    idx = min(int(p * 10), 9)  # 0..9
    low = idx / 10
    high = (idx + 1) / 10

    labels = [
        "very safe",
        "safe",
        "mostly safe",
        "slightly safe",
        "borderline",
        "some risk",
        "moderate risk",
        "high risk",
        "very high risk",
        "extremely high risk",
    ]
    return f"{low:0.2f} - {high:0.2f}", labels[idx]


# ---------------------------
# Geo + Hospital lookup helpers
# ---------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_location_via_ip():
    """
    Reliable fallback location using public IP.
    No permissions required.
    """
    r = requests.get("https://ipinfo.io/json", timeout=10)
    r.raise_for_status()
    data = r.json()

    loc = data.get("loc")  # "lat,lon"
    if not loc:
        return None, None, "Unknown location"

    lat, lon = loc.split(",")

    city = data.get("city", "")
    region = data.get("region", "")
    country = data.get("country", "")

    label = f"{city}, {region}, {country}"

    return float(lat), float(lon), label

@st.cache_data(show_spinner=False)
def reverse_geocode(lat, lon):
    """
    Convert (lat,lon) to an approximate human-readable address using OSM Nominatim.
    Cached to reduce requests.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "jsonv2", "lat": lat, "lon": lon}
    headers = {"User-Agent": "ECS171-HeartDemo/1.0 (educational project)"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("display_name", "Unknown location")


@st.cache_data(show_spinner=False)
def find_nearby_hospitals(lat, lon, radius_m=5000, limit=15):
    """
    Query Overpass API for hospitals/clinics around (lat,lon).
    Uses fallback Overpass servers if one is down/busy.
    """
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    ]

    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});

      node["amenity"="clinic"](around:{radius_m},{lat},{lon});
      way["amenity"="clinic"](around:{radius_m},{lat},{lon});
      relation["amenity"="clinic"](around:{radius_m},{lat},{lon});
    );
    out center;
    """

    last_err = None
    data = None

    for url in overpass_urls:
        try:
            r = requests.post(url, data=query, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            continue

    if data is None:
        raise RuntimeError(f"All Overpass servers failed. Last error: {last_err}")

    results = []
    for el in data.get("elements", []):
        if "lat" in el and "lon" in el:
            el_lat, el_lon = el["lat"], el["lon"]
        else:
            center = el.get("center")
            if not center:
                continue
            el_lat, el_lon = center["lat"], center["lon"]

        tags = el.get("tags", {})
        name = tags.get("name", "(Unnamed)")
        amenity = tags.get("amenity", "hospital/clinic")

        results.append({"name": name, "amenity": amenity, "lat": el_lat, "lon": el_lon})

    # Deduplicate + distance + sort
    seen = set()
    uniq = []
    for x in results:
        key = (x["name"], round(x["lat"], 6), round(x["lon"], 6))
        if key not in seen:
            seen.add(key)
            x["distance_km"] = haversine_km(lat, lon, x["lat"], x["lon"])
            uniq.append(x)

    uniq.sort(key=lambda d: d["distance_km"])
    return uniq[:limit]


# ---------------------------
# App UI
# ---------------------------
st.title("Heart Disease Risk Prediction")
st.caption("Educational demo only — not medical advice or a diagnosis.")
st.caption("Team 9 - Zhichu Zheng, Alexander Davis, Tianyuan Fu")

# Load model
try:
    pipeline, selected_features = load_artifact()
except Exception as e:
    st.error(f"Could not load model artifact at `{MODEL_PATH}`.")
    st.code(str(e))
    st.stop()


# Explanations
with st.expander("What do these factors mean?"):
    st.markdown(
        """
**Age**: patient age in years.  
**Sex**: Female/Male.  

**Chest pain type (cp)**:
- 1 = Typical angina
- 2 = Atypical angina
- 3 = Non-anginal pain
- 4 = Asymptomatic  

**Max heart rate achieved (thalach)**: highest heart rate during exercise test.

**Oldpeak**: ST depression induced by exercise relative to rest. Higher often indicates more abnormality.

**Exercise-induced angina (exang)**: 0 = No, 1 = Yes.

**Number of major vessels (ca)**: integer 0–4 vessels colored by fluoroscopy.

**Thal (thal)** (common UCI-style):
- 3 = Normal
- 6 = Fixed defect
- 7 = Reversible defect  
*(Your model uses `thal_7.0`, i.e., whether thal is 7.)*
        """
    )

with st.expander("Risk scale legend (probability → label)"):
    labels = [
        "very safe",
        "safe",
        "mostly safe",
        "slightly safe",
        "borderline",
        "some risk",
        "moderate risk",
        "high risk",
        "very high risk",
        "extremely high risk",
    ]
    rows = [(f"{i/10:0.2f} - {(i+1)/10:0.2f}", labels[i]) for i in range(10)]
    st.table(pd.DataFrame(rows, columns=["Probability range", "Risk label"]))


# ---------------------------
# Sample patients (for demo)
# ---------------------------
st.divider()
st.subheader("Patients Prediction")

# Initialize defaults in session_state (only once)
defaults = {
    "age": 45,
    "sex_label": "Male",
    "cp_choice": "4 — Asymptomatic",
    "thalach": 150,
    "oldpeak": 1,
    "exang_label": "0 — No",
    "ca": 0,
    "thal_choice": "3 — Normal",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# Two demo samples tuned to your 8-feature model
HIGH_RISK_SAMPLE = {
    "age": 62,
    "sex_label": "Male",
    "cp_choice": "4 — Asymptomatic",      # cp_4 = 1
    "thalach": 110,                        # low max HR
    "oldpeak": 3,                         # higher ST depression
    "exang_label": "1 — Yes",             # exang = 1
    "ca": 2,                              # more vessels
    "thal_choice": "7 — Reversible defect" # thal_7.0 = 1
}

HEALTHY_SAMPLE = {
    "age": 48,
    "sex_label": "Male",
    "cp_choice": "1 — Typical angina",    # cp_4 = 0
    "thalach": 175,                       # higher max HR
    "oldpeak": 0,
    "exang_label": "0 — No",
    "ca": 0,
    "thal_choice": "3 — Normal"
}

c1, c2 = st.columns(2)
with c1:
    if st.button("Load High-Risk Sample", use_container_width=True):
        st.session_state.update(HIGH_RISK_SAMPLE)
        st.success("High-risk sample loaded. Scroll down and click Predict.")

with c2:
    if st.button("Load Healthy Sample", use_container_width=True):
        st.session_state.update(HEALTHY_SAMPLE)
        st.success("Healthy sample loaded. Scroll down and click Predict.")

# ---------------------------
# Patient input form
# ---------------------------
with st.form("patient_form"):
    st.subheader("Enter patient info")

    age = st.selectbox("Age (years)", list(range(1, 121)), key="age")

    sex_label = st.selectbox("Sex", ["Female", "Male"], key="sex_label")

    cp_map = {
        "1 — Typical angina": 1,
        "2 — Atypical angina": 2,
        "3 — Non-anginal pain": 3,
        "4 — Asymptomatic": 4,
    }
    cp_choice = st.selectbox("Chest pain type (cp)", list(cp_map.keys()), key="cp_choice")
    cp = cp_map[cp_choice]

    thalach = st.selectbox("Max heart rate achieved (thalach)", list(range(60, 221)), key="thalach")

    oldpeak = st.selectbox("Oldpeak (integer for demo)", list(range(0, 11)), key="oldpeak")

    exang_label = st.selectbox("Exercise-induced angina (exang)", ["0 — No", "1 — Yes"], key="exang_label")
    exang = 1 if exang_label.startswith("1") else 0

    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4], key="ca")

    thal_map = {
        "3 — Normal": 3,
        "6 — Fixed defect": 6,
        "7 — Reversible defect": 7,
    }
    thal_choice = st.selectbox("Thal (thal)", list(thal_map.keys()), key="thal_choice")
    thal = thal_map[thal_choice]

    submitted = st.form_submit_button("Predict")


# ---------------------------
# Prediction result
# ---------------------------
if submitted:
    # Convert to the EXACT feature vector your trained model expects:
    # ['cp_4','ca','thal_7.0','exang','oldpeak','thalach','sex','age']
    sex = 1 if sex_label == "Male" else 0
    cp_4 = 1 if cp == 4 else 0
    thal_7 = 1 if thal == 7 else 0

    X = pd.DataFrame([{
        "cp_4": cp_4,
        "ca": float(ca),
        "thal_7.0": thal_7,
        "exang": int(exang),
        "oldpeak": float(oldpeak),
        "thalach": float(thalach),
        "sex": int(sex),
        "age": float(age),
    }])[selected_features]

    try:
        proba = float(pipeline.predict_proba(X)[0][1])
        pred = int(proba >= 0.5)
        bucket_range, bucket_label = risk_bucket(proba)

        st.divider()
        st.subheader("Result")

        st.write(f"Predicted probability of heart disease: **{proba:.2f}**")
        st.write(f"Risk scale: **{bucket_range} → {bucket_label}**")
        st.progress(max(0.0, min(1.0, proba)))

        if pred == 1:
            st.error("Model prediction: **Heart Disease (higher risk)**")
        else:
            st.success("Model prediction: **No Heart Disease (lower risk)**")

        st.caption("Threshold = 0.50 for demo. Educational use only.")
    except Exception as e:
        st.error("Prediction failed (feature mismatch or model artifact issue).")
        st.code(str(e))


# ---------------------------
# Location + Hospitals section (GPS first, fallback to IP; address-only UI + hospital addresses)
# ---------------------------
st.divider()
st.subheader("Nearby Hospitals")
st.caption("We only use your location to show nearby hospitals. No data is stored.")

# ---- Session state ----
st.session_state.setdefault("lat", None)
st.session_state.setdefault("lon", None)
st.session_state.setdefault("address", None)
st.session_state.setdefault("hospitals", None)
st.session_state.setdefault("gps_clicks", 0)

# ---- Short reverse geocode for hospitals (cached) ----
@st.cache_data(show_spinner=False)
def reverse_geocode_short(lat, lon):
    """
    Reverse geocode to a shorter, user-friendly address string.
    Cached to reduce requests.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 1}
    headers = {"User-Agent": "ECS171-HeartDemo/1.0 (educational project)"}

    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    a = data.get("address", {})

    road = a.get("road") or a.get("pedestrian") or a.get("footway") or ""
    house = a.get("house_number") or ""
    city = a.get("city") or a.get("town") or a.get("village") or ""
    state = a.get("state") or ""
    postcode = a.get("postcode") or ""

    street = (f"{house} {road}").strip()
    parts = [p for p in [street, city, state, postcode] if p]
    return ", ".join(parts) if parts else data.get("display_name", "Address unavailable")

# ---- Controls (km) ----
col1, col2 = st.columns(2)
with col1:
    radius_km = st.selectbox("Search radius (km)", [1, 2, 5, 10, 20, 30], index=2)
with col2:
    max_results = st.selectbox("Max results", [5, 10, 15, 20], index=2)

# ---- Get your address (GPS first -> IP fallback) ----
if st.button("Get your address", key="btn_get_address"):
    st.session_state["hospitals"] = None  # clear previous results

    # 1) Try GPS first
    coords = streamlit_js_eval(
        js_expressions="""
        new Promise((resolve) => {
            if (!navigator.geolocation) { resolve(null); return; }
            navigator.geolocation.getCurrentPosition(
                (pos) => resolve({lat: pos.coords.latitude, lon: pos.coords.longitude}),
                () => resolve(null),
                { enableHighAccuracy: true, timeout: 10000 }
            );
        })
        """,
        want_output=True,
        key=f"gps_addr_{st.session_state['gps_clicks']}"
    )
    st.session_state["gps_clicks"] += 1

    if coords:
        # GPS success
        st.session_state["lat"] = float(coords["lat"])
        st.session_state["lon"] = float(coords["lon"])
        try:
            st.session_state["address"] = reverse_geocode(st.session_state["lat"], st.session_state["lon"])
        except Exception:
            st.session_state["address"] = "Address unavailable (reverse geocoding failed)."
    else:
        # 2) GPS failed -> IP fallback
        try:
            lat, lon, label = get_location_via_ip()
            if lat is None or lon is None:
                st.session_state["lat"] = None
                st.session_state["lon"] = None
                st.session_state["address"] = None
                st.warning("Could not determine your location (GPS blocked and IP lookup failed).")
            else:
                st.session_state["lat"] = float(lat)
                st.session_state["lon"] = float(lon)

                # Prefer reverse-geocoded address, fallback to ip label
                try:
                    st.session_state["address"] = reverse_geocode(st.session_state["lat"], st.session_state["lon"])
                except Exception:
                    st.session_state["address"] = label
        except Exception as e:
            st.session_state["lat"] = None
            st.session_state["lon"] = None
            st.session_state["address"] = None
            st.error("Address lookup failed.")
            st.code(str(e))

# ---- Display address only (no lat/lon) ----
if st.session_state["address"]:
    st.write(f"**Approx. address:** {st.session_state['address']}")
else:
    st.info("Click **Get your address** to enable nearby hospital lookup.")

# ---- Find nearby hospitals (requires lat/lon internally, but we won't display them) ----
if st.session_state["lat"] is not None and st.session_state["lon"] is not None:
    if st.button("Find nearby hospitals", key="btn_find_hospitals"):
        radius_m = int(radius_km * 1000)
        with st.spinner(f"Searching within {radius_km} km..."):
            try:
                st.session_state["hospitals"] = find_nearby_hospitals(
                    st.session_state["lat"],
                    st.session_state["lon"],
                    radius_m=radius_m,
                    limit=max_results
                )
            except Exception as e:
                st.session_state["hospitals"] = []
                st.error("Hospital lookup failed (Overpass API might be busy). Try again or change radius.")
                st.code(str(e))

# ---- Render hospitals with addresses (no user lat/lon shown) ----
if st.session_state["hospitals"] is not None:
    hospitals = st.session_state["hospitals"]

    if len(hospitals) == 0:
        st.warning(f"No hospitals/clinics found within {radius_km} km.")
    else:
        st.markdown("### Nearby hospitals/clinics")

        df_h = pd.DataFrame(hospitals)
        df_h["distance_km"] = df_h["distance_km"].map(lambda x: round(float(x), 2))

        # Add hospital address column (reverse geocode only top N to reduce API load)
        df_h["address"] = "—"
        TOP_N_ADDR = min(len(df_h), 10)
        df_h.loc[:TOP_N_ADDR-1, "address"] = df_h.loc[:TOP_N_ADDR-1].apply(
            lambda row: reverse_geocode_short(row["lat"], row["lon"]),
            axis=1
        )

        # Optional map: shows points only (no coordinate text)
        st.map(df_h[["lat", "lon"]])

        # Table: show name + distance + address
        st.dataframe(
            df_h[["name", "amenity", "distance_km", "address"]],
            use_container_width=True
        )

        top = df_h.iloc[0]
        st.info(f"Closest: **{top['name']}** — {top['distance_km']:.2f} km — {top['address']}")