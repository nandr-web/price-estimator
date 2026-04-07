FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src"

COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]
