backend:
	cd server && python3 -m venv venv && \
	source venv/bin/activate && \
	make

react-app:
	cd frontend &&\
	npm start

