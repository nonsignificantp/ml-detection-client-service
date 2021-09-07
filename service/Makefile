build:
	@echo "=========== Updating image ==========="
	@docker rmi app/api:latest || exit 0
	@docker build -t app/api .

run:
	@echo "=========== Starting server ==========="
	docker run -it --rm \
		-p 8081:8080 \
		-v ${PWD}/app:/app \
		--env-file .env \
		--network devel-nw \
		app/api:latest \
		/start-reload.sh

service: build run