FROM public.ecr.aws/lambda/python:3.11

RUN yum install -y gcc gcc-c++ cmake3 && \
    ln -sf /usr/bin/cmake3 /usr/bin/cmake && \
    yum clean all

COPY requirements-lambda.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --only-binary :all: \
      "pandas>=2.0,<3" \
      "numpy>=1.26,<2" \
      "scikit-learn>=1.4,<2" \
      "xgboost>=2.0,<3" \
      "shap>=0.44,<1" \
      "rapidfuzz>=3.0,<4" \
      "fastapi>=0.110,<1" \
      "pydantic>=2.0,<3" \
      "joblib>=1.3,<2" \
      "mangum>=0.17,<1" \
      "boto3>=1.34,<2" && \
    pip install --no-cache-dir "lightgbm>=4.0,<5"

COPY src/ ${LAMBDA_TASK_ROOT}/src/
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src"

COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]
