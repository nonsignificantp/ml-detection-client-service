build:
	@docker rmi app/client || exit 0
	@docker build -t app/client .

run:
	@docker run -it --rm \
		-p 8080:8080 \
		-v ${PWD}/app:/app \
		app/client \

client: build run