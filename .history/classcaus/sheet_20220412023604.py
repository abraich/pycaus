import os 
from google import create_Service

CILENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'sheets'
API_VERSION = 'v4'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

service = Create_Service(CILENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

spreadsheet_id = 'sq8rXYLa3FhHYK1W3ZbIr1lZ06597h5IqRFIi8Q'
myspreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()


