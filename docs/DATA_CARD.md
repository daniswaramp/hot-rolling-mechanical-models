# Data Card: Industrial Steel Rolling Mechanical Testing Datasets
## Heavy Plate Sequential Rolling & Steckel Mill Hot Rolled Coil (HRC)

---

## 1. Dataset Motivation & Origin

- **Data Provider**: PT Gunung Raja Paksi Tbk (Cikarang Barat, West Java, Indonesia).
- **Domain**: Hot-rolled structural, pressure vessel, and offshore plate and coil manufacturing.
- **Collection Period**: Industrial operational rolling campaigns through 2026.
- **Primary Purpose**: Predictive mechanical modeling, online quality assurance, alloy cost optimization, and specification compliance forecasting.

---

## 2. Dataset Composition & Scale

The data platform integrates two distinct industrial production lines:

| Dataset Dimension | Heavy Plate Mill (`Plate/`) | Steckel Hot Strip Mill (`HRC/`) |
| :--- | :--- | :--- |
| **Source Log File** | `PLATE SEQUENTIAL ROLLING RESULT - 2026.xlsx` | `STECKEL_MILL_REPORT.xlsx` |
| **Total Ingested Records** | **23,845 plates** | **20,298 coils** |
| **Raw Column Count** | 84 process & test variables | 35 process & test variables |
| **Unique Melts (Ladle Heats)** | 1,765 heats | 4,595 heats |
| **Gauge Range** | $6.0\text{ mm} - 205.0\text{ mm}$ (99% $\le 70.0\text{ mm}$) | $2.5\text{ mm} - 16.0\text{ mm}$ |
| **Width Range** | $1,240\text{ mm} - 3,160\text{ mm}$ | $900\text{ mm} - 1,880\text{ mm}$ |
| **Primary Steel Grades** | SS400, ASTM A 36, SM490B, ASME SA 516 Gr 70 | SPHC, SS400, SM490A, ASTM A 1011 |

---

## 3. Data Schema & Feature Categories

Each record captures the complete lifecycle of a slab from liquid metallurgy to room-temperature mechanical testing:

1. **Ladle Melt Chemistry (Optical Emission Spectrometry - OES)**:
   - Base elements: Carbon ($C$), Manganese ($Mn$), Silicon ($Si$), Phosphorus ($P$), Sulfur ($S$).
   - Solution & microalloy elements: Copper ($Cu$), Nickel ($Ni$), Chromium ($Cr$), Molybdenum ($Mo$), Vanadium ($V$), Niobium ($Nb$), Titanium ($Ti$), Aluminum ($Al$), Nitrogen ($N$), Boron ($B$).
2. **Upstream Geometry**:
   - Initial slab thickness ($Slab\_T$), slab width, slab length.
   - Heavy plate transfer bar thickness ($TB\_Thick$), intermediate roughing draft.
   - Steckel intermediate transfer bar gauge ($TB\_Thick\_mm$).
3. **Thermomechanical Processing (TMCP)**:
   - Finishing mill entry temperature, Finish Rolling Temperature ($FRT$ or $FT$).
   - Heavy Plate Accelerated Cooling Header count ($N_{headers}$), laminar flow activation, and Post-Laminar water exit temperature ($T_{post}$).
   - Steckel Runout Table (ROT) top and bottom coiling spray temperatures ($CT_{top}, CT_{bot}$), rolling speed.
4. **Mechanical Testing Targets (Universal Testing Machine - UTM & Charpy Pendulum)**:
   - `YS_BN_MPa`: Upper/Lower Yield Strength (As-Rolled / TMCP before normalizing).
   - `TS_BN_MPa`: Ultimate Tensile Strength (As-Rolled / TMCP before normalizing).
   - `EL_BN_Pct`: Plastic Elongation percentage (gauge length $L_0 = 5.65\sqrt{S_0}$ or 200 mm).
   - `YS_AN_MPa`, `TS_AN_MPa`, `EL_AN_Pct`: Tensile properties after furnace normalizing ($910-930^\circ\text{C}$).
   - `Impact_0C_J`, `Impact_m20C_J`, `Impact_m40C_J`: Charpy V-Notch impact energy absorbing capacity.
   - `Z_Test_Pct`: Through-thickness reduction of area (Z-direction resistance to lamellar tearing).

---

## 4. Preprocessing, Audit Trail & Cleaning Rules

Raw manufacturing logs contain known human data-entry artifacts and instrumentation noise. Automated ETL applies deterministic physical cleaning rules:

1. **Fast Pure-Python Extraction**:
   - Ingested via `python-calamine` (Rust-based Excel engine), parsing 24,000+ rows in ~8-12 seconds without COM overhead.
2. **Decimal Point Omission Typo Auto-Correction**:
   - Typo detection: Manual operators occasionally omit decimal points during manual entry (e.g. typing `41205` instead of `412.05 MPa`).
   - Automatically detected and rescaled: if $val > \text{bound}_{max}$ and $val / 100 \in [\text{bound}_{min}, \text{bound}_{max}]$, value is safely divided by 100.
3. **Laminar Header & Column Swap Parsing**:
   - In Heavy Plate sheets, operators occasionally swap header counts and pyrometer temperatures (e.g. entering `750` in `ACC_Laminar_Header` column).
   - Regex and range parsers disentangle header text (`"2H"`, `"H4"`) and restore misallocated temperatures.
4. **Physical Bounds Enforcement**:
   - Enforces $YS < TS$ consistency; records with impossible strength inversions ($YS \ge TS$) are flagged and nullified.
   - Enforces $Coil\_T < TB\_Thick$ (positive reduction drafting).
5. **Audit Logging**:
   - All modified, rescaled, or nullified cells are exported to `Plate/reports/cleaning_log.csv` and `HRC/reports/hrc_cleaning_log.csv`.

---

## 5. Melt-Grouped Data Splitting (Zero Leakage)

- **Grouping Key**: `Heat_ID` (Ladle Melt Identification).
- **Mechanism**: `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`.
- **Integrity Assertion**:
  ```python
  train_heats = set(train_df["Heat_ID"].unique())
  val_heats = set(val_df["Heat_ID"].unique())
  assert len(train_heats.intersection(val_heats)) == 0
  ```
- **Validation Scale**:
  - Heavy Plate: 1,632 held-out validation plates from 353 independent heats.
  - Steckel Mill: 2,221 held-out validation coils from 919 independent heats.

---

## 6. Subfolder Isolation & Privacy

- Proprietary operational campaign workbooks reside strictly in local root workspace (`Plate/raw/`, `HRC/raw/`).
- The public GitHub repository mirror (`push-to-GH/`) contains only sanitized features, pre-trained model binaries, documentation, and the interactive Streamlit application.
- Syncing between workspace and GitHub is managed strictly through `run_github_sync.bat`.
