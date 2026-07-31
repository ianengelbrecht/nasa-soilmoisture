## Extracting soil moisture data as tif files from NASA AppEEARS service

We just need a geojson boundary in WGS84 projection (rename accordingly this in submit_appeears.py).
Also add a .env file with AppEEARS / NASA Earthdata login details.

Soil moisture is available up to 3 days before now. 

Process is as follows:
- Submit request for single date or date range with `submit_appeears.py`
- Request reciept confirmation email will be sent.
- Dataset ready email will be sent (typically just a few minutes). 
- Fetch the data with `download_appeears.py`
- Create a video to check for changes using `create_video.py`
- Import files into QGIS and style as needed (use viridis colour ramp with scale of 0.0 to 0.4)

All scripts can be used with `-h` for assistance.

