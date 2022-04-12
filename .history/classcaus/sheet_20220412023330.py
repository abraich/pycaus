import os 
from Google import Create_Service

CILENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'sheets'
API_VERSION = 'v4'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

service = Create_Service(CILENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

spreadsheet_id = '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms'
