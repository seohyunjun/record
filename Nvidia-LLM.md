
# NVIDIA LLM

### AICA 23년 AI 데이터 센터 지원 사업 소개 
<br>

#### 인공지능산업 융합사업단_박종선 책임

- AI 서버 제공
- 지원 대상 : 
    - AI 목적 기업, 광주 기업 추가 점수
    - GPU : A100, T4
    - H100 지원 여부 (지원 9월 부터 예상  )
    - 2월 신청 예정
    - 신청서, 기획서를 통해 선정
    - 개인 정보 동의서 etc 제출
    - 3월~ 내년 3월/ 1년 단위
    - 우수 성과, 데이터 공유 시 지원 기업 추가 지원
    - AI 데이터 센터 규모 증가 
    - 하반기 클라우드 서비스 제공
    - [인공지능 사업단](http://www.aica-gj.kr/sub.php?PID=0401)

<br>

### Nemo-Megatron 
- chatGPT 인기
- GENERATIVE AI (X->Y)
- LARGE LANGUAGE MODEL(LLM) WITH TRANSFORMER
- LLM 모델이 왜 학습이 어려운지?
    - Data Paralleism
        - training speed
        - different data
        - all workers have the same copy of the model
        - Neural network gradients are exchanged
    - Model Paralleism
        - Allows you to use a bigger model
        - All workers train on the same data
        - 모델 쪼개기
        - Pipeline paralleism []
        - Tensor paralleism ~
        - 메가트론-LM 
            - Pipeline Parallelism 
            - NeMo Megatron 
                - large language 모델 train을 위한 프레임워크
            - 오픈 모델
            - Nvidia 내부에서 사용 
        - NeMo-Megatron
            - 범용
            - HP TOOL
                - Batch Size, Parameter Setting 용이
        - FT + Triton
            - TensorRT vs FasterTransformer
                - FasterTransformer model parel에 사용
                - 일반적으로 TensorRT 사용
                - FT 두개의 서버를 하나로 사용 가능
                - FT converter 지원 (Megatron, Huggingface, ONNX)
                - Triton 앙상블로 (Tokenizing, FT, Detokenizing 지원)

### NVIDIA NeMo Megatron 실습

- [nlp-containers](ngc.nvidia.com/containers/ea-bignlp:bignlp-trainig)
- yaml로 정의
- training-prompt_learning
- cluster 작성 후 model 학습 (LLM)

- HP tool
    - gpt tool
        - 학습 파라미터 자동 설정
            1. config yaml 작성 (path)
            2. size 설정 (unkown_size default)
            3. run

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

### Riva Solution Update Demo (Rova with Korean ASR)

소우진님 (Woojin Soh wsoh@nvidia.com)

- [2022 meet-up](https://developer.nvidia.com/ko-kr/blog/2022%eb%85%84-nvidia%ec%97%90%ec%84%9c-%ec%a4%80%eb%b9%84%ed%95%9c-%ea%b8%b0%ec%88%a0-%eb%8d%b0%eb%aa%a8-%ec%84%b8%ec%85%98-%ec%b4%9d%ec%a0%95%eb%a6%ac/)
- riva 
    - 커뮤니티 버전 무료 
    - Enterprise (sdk / 구축 지원 서비스)
    - riva studio 하반기 제공 예정
        - GUI를 통한 tts 모델 생성
    - [riva demo](https://github.com/woojinsoh/riva_demo)
        - 5번 주피터 파일
        - rmir(확장자) : pipeline 수행 파일
 
