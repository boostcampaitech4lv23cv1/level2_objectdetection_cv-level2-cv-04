#Streamlit Visualization
##Ground Truch 가 존재하는 파일에 대한 inference 결과를 시각화합니다.

###how to start
```
pip install streamlit
```

###modify
app.py 내부의
1. submission_file_dir = #inference 한 submission.csv 파일 경로
2. gt_json_file_dir = #ground truth 의 메타데이터 json 파일 경로
3. dataset_path = '/opt/ml/dataset/' #전체 데이터가 존재하는 경로
를 수정해주세요

###run
```
streamlit run app.py --server.fileWatcherType none --server.port=30005
```