import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import pandas as pd
import joblib
import math
import requests

MODEL_PATH = "model/model.joblib"

st.set_page_config(page_title="Heart Disease Risk Demo", layout="centered")


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
st.title("Heart Disease Risk Prediction (Demo)")
st.caption("Educational demo only — not medical advice or a diagnosis.")


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
# Patient input form
# ---------------------------
with st.form("patient_form"):
    st.subheader("Enter patient info")

    age = st.selectbox("Age (years)", list(range(1, 121)), index=44)  # default 45
    sex_label = st.selectbox("Sex", ["Female", "Male"])

    cp_map = {
        "1 — Typical angina": 1,
        "2 — Atypical angina": 2,
        "3 — Non-anginal pain": 3,
        "4 — Asymptomatic": 4,
    }
    cp_choice = st.selectbox("Chest pain type (cp)", list(cp_map.keys()))
    cp = cp_map[cp_choice]

    thalach = st.selectbox("Max heart rate achieved (thalach)", list(range(60, 221)), index=90)  # default ~150

    oldpeak = st.selectbox("Oldpeak (integer for demo)", list(range(0, 11)), index=1)

    exang_label = st.selectbox("Exercise-induced angina (exang)", ["0 — No", "1 — Yes"])
    exang = 1 if exang_label.startswith("1") else 0

    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4], index=0)

    thal_map = {
        "3 — Normal": 3,
        "6 — Fixed defect": 6,
        "7 — Reversible defect": 7,
    }
    thal_choice = st.selectbox("Thal (thal)", list(thal_map.keys()))
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
# Location + Hospitals section
# ---------------------------
st.divider()
st.subheader("Nearby Hospitals")
st.caption("If you allow location access, we only use it to show nearby hospitals. No data is stored.")

# Init session state (persistent across reruns)
st.session_state.setdefault("lat", None)
st.session_state.setdefault("lon", None)
st.session_state.setdefault("loc_source", None)
st.session_state.setdefault("address", None)
st.session_state.setdefault("hospitals", None)  # store lookup results


# ---- Settings for hospital lookup (km) ----
col1, col2 = st.columns(2)
with col1:
    radius_km = st.selectbox("Search radius (km)", [1, 2, 5, 10, 20, 30], index=2)
with col2:
    max_results = st.selectbox("Max results", [5, 10, 15, 20], index=2)


# ---- Location buttons ----
colA, colB = st.columns(2)

with colA:
    if st.button("Use GPS location (recommended)", key="btn_gps"):
        coords = streamlit_js_eval(
            js_expressions="""
            new Promise((resolve) => {
                navigator.geolocation.getCurrentPosition(
                    (pos) => resolve({lat: pos.coords.latitude, lon: pos.coords.longitude}),
                    () => resolve(null),
                    { enableHighAccuracy: true, timeout: 8000 }
                );
            })
            """,
            want_output=True,
            key=f"gps_{st.session_state.get('gps_clicks', 0)}"
        )
        st.session_state["gps_clicks"] = st.session_state.get("gps_clicks", 0) + 1

        if coords:
            st.session_state["lat"] = float(coords["lat"])
            st.session_state["lon"] = float(coords["lon"])
            st.session_state["loc_source"] = "GPS"
            st.session_state["hospitals"] = None  # clear old results
            st.success("Using precise GPS location.")
        else:
            st.warning("GPS unavailable — using approximate IP location instead.")
            try:
                lat, lon, label = get_location_via_ip()
                st.session_state["lat"] = float(lat)
                st.session_state["lon"] = float(lon)
                st.session_state["loc_source"] = f"IP: {label}"
                st.session_state["hospitals"] = None
                st.success(f"Approximate location detected: {label}")
            except Exception as e:
                st.error("Could not determine location.")
                st.code(str(e))

with colB:
    if st.button("Use approximate location by IP (no permission)", key="btn_ip"):
        try:
            lat, lon, label = get_location_via_ip()
            if lat is None or lon is None:
                st.warning("IP location failed to return coordinates.")
            else:
                st.session_state["lat"] = float(lat)
                st.session_state["lon"] = float(lon)
                st.session_state["loc_source"] = f"IP: {label}"
                st.session_state["hospitals"] = None
                st.success(f"Approximate location detected: {label}")
        except Exception as e:
            st.error("IP location lookup failed.")
            st.code(str(e))


# ---- Show location + add hospital lookup button ----
if st.session_state["lat"] is not None and st.session_state["lon"] is not None:
    lat, lon = st.session_state["lat"], st.session_state["lon"]
    st.success(f"Location set via: {st.session_state['loc_source']}")
    st.write(f"**Latitude:** {lat:.5f}  |  **Longitude:** {lon:.5f}")

    # Optional: reverse geocode address (nice for demo)
    if st.button("Show approximate address", key="btn_address"):
        try:
            st.session_state["address"] = reverse_geocode(lat, lon)
        except Exception as e:
            st.session_state["address"] = None
            st.warning("Could not reverse-geocode address.")
            st.code(str(e))

    if st.session_state["address"]:
        st.write(f"**Approx. address:** {st.session_state['address']}")

    # Hospital lookup
    if st.button("Find nearby hospitals/clinics", key="btn_find_hospitals"):
        radius_m = int(radius_km * 1000)
        with st.spinner(f"Searching within {radius_km} km..."):
            try:
                hospitals = find_nearby_hospitals(lat, lon, radius_m=radius_m, limit=max_results)
                st.session_state["hospitals"] = hospitals
            except Exception as e:
                st.session_state["hospitals"] = []
                st.error("Hospital lookup failed (Overpass API might be busy). Try again or change radius.")
                st.code(str(e))

    # ---- Always render results if available ----
    if st.session_state["hospitals"] is not None:
        hospitals = st.session_state["hospitals"]

        if len(hospitals) == 0:
            st.warning(f"No hospitals/clinics found within {radius_km} km.")
        else:
            st.markdown("### 🏥 Nearby hospitals/clinics")

            df_h = pd.DataFrame(hospitals)
            df_h["distance_km"] = df_h["distance_km"].map(lambda x: round(float(x), 2))

            # Map: user + hospitals
            user_df = pd.DataFrame([{"lat": lat, "lon": lon}])
            hosp_df = df_h[["lat", "lon"]]
            st.map(pd.concat([user_df, hosp_df], ignore_index=True))

            # Table list
            st.dataframe(
                df_h[["name", "amenity", "distance_km", "lat", "lon"]],
                use_container_width=True
            )

            # Closest facility
            top = hospitals[0]
            st.info(f"Closest: **{top['name']}** ({top['distance_km']:.2f} km)")

else:
    st.info("Choose GPS or IP-based location to enable nearby hospital lookup.")