# Workflow for obtaining sparta data from NRLMSIS-00 model:
  1. gfz_cleaner.py            → space_weather_daily_1932_2026.csv (PROVIDED ABOVE)
  2. nrlmsis_runner.py         → output_daily.csv   ← this script reads it (Could not provide as file is too big for Github)
  3. run_msis_analog_predictor.py    → msis_analog_predictor.pkl
  4. Use pkl for queries to output the SPARTA inputs

