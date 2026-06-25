# Use NRLMSIS-00 for SPARTA inputs, DTM-2020 for finding total mass density
# Workflow for obtaining sparta data from NRLMSIS-00 model:
  1. gfz_cleaner.py            → space_weather_daily_1932_2026.csv (PROVIDED ABOVE)
  2. nrlmsis_runner.py         → output_daily.csv   ← this script reads it (Could not provide as file is too big for Github)
  3. run_msis_analog_predictor.py  (MUST HAVE THE msis_dtm_analog_predictor file in the SAME FOLDER)  → msis_analog_predictor.pkl
  4. Use pkl for queries to output the SPARTA inputs
# Workflow for obtaining sparta data from DTM-2020 model:
  1. Upload: run_dtm_analog_predictor.py,    msis_dtm_analog_predictor.py,      mcm_results_all_altitudes.csv        onto ubuntu in the same file folder
  2. Follow instructions on run_dtm_analog_predictor.py (make sure you edit the user paths)
