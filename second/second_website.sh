python ./kittiviewer/backend/main.py main --port=50001 > /dev/null &
cd ./kittiviewer/frontend && python -m http.server
