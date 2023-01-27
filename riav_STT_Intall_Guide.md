## Nvidia Riva install guide (Speech to Text)

<img src="https://github.com/seohyunjun/record/blob/main/source/riva_demo_env.png?raw=true" title="rive demo env" width=90%>

<br>

# Service-maker로 원하는 모델 생성 

## ngc 등록
1. [ngc](https://catalog.ngc.nvidia.com/) 가입
2. nvcr.io에 API 등록
```
docker login nvcr.io
# Username: $oauthtoken
# Password: [ngc API KEY]
 ```

### service-maker로 원하는 모델 생성 (STT)
1. git clone [riva demo](https://github.com/woojinsoh/riva_demo) 

2. ngc pull riva_quickstart 
```
ngc registry resource download-version "nvidia/riva/riva_quickstart:2.8.1"
```

3. riva network set
```
docker network create riva-speech
```

4. config 파일 수정
```
asr_acoustic_model=citrinet_1024
```

5. 한국어 speech model 다운로드

[speechtotext_ko_kr_lm](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_ko_kr_lm)
[speechtotext_ko_kr_citrinet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_ko_kr_citrinet/files)
```
mkdir servicemaker-dev/models/korean #안에 추가 
```

6. riva-service-maker 구동& volume 추가 
```
bash riva_init.sh 
```
```
bash riva_start.sh # docker volume (models/korean) 추가
```

7. service-maker container에서 model 생성
```
#sh 파일 안의 volume 위치를 customize한 뒤 build-deploy 진행
scripts/build_deploy/korean_models/riva_asr_citrinet_kr_build.sh
scripts/build_deploy/korean_models/riva_asr_citrinet_kr_deploy.sh
```

8. riva-start-client 실행 후 주피터로 테스트