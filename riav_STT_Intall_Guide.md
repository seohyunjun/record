## Nvidia Riva install guide (Speech to Text)

<img src="https://github.com/seohyunjun/record/blob/main/source/riva_demo_env.png?raw=true" title="rive demo env" width=90%>

<br>

# Service-maker�� ���ϴ� �� ���� 

## ngc ���
1. [ngc](https://catalog.ngc.nvidia.com/) ����
2. nvcr.io�� API ���
```
docker login nvcr.io
# Username: $oauthtoken
# Password: [ngc API KEY]
 ```

### service-maker�� ���ϴ� �� ���� (STT)
1. git clone [riva demo](https://github.com/woojinsoh/riva_demo) 

2. ngc pull riva_quickstart 
```
ngc registry resource download-version "nvidia/riva/riva_quickstart:2.8.1"
```

3. riva network set
```
docker network create riva-speech
```

4. config ���� ����
```
asr_acoustic_model=citrinet_1024
```

5. �ѱ��� speech model �ٿ�ε�

[speechtotext_ko_kr_lm](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_ko_kr_lm)
[speechtotext_ko_kr_citrinet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_ko_kr_citrinet/files)
```
mkdir servicemaker-dev/models/korean #�ȿ� �߰� 
```

6. riva-service-maker ����& volume �߰� 
```
bash riva_init.sh 
```
```
bash riva_start.sh # docker volume (models/korean) �߰�
```

7. service-maker container���� model ����
```
#sh ���� ���� volume ��ġ�� customize�� �� build-deploy ����
scripts/build_deploy/korean_models/riva_asr_citrinet_kr_build.sh
scripts/build_deploy/korean_models/riva_asr_citrinet_kr_deploy.sh
```

8. riva-start-client ���� �� �����ͷ� �׽�Ʈ