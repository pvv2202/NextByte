backend:
	cd server && \
	python3 -m venv venv &&  source venv/bin/activate && \
	python3 server.py

react-app:
	cd frontend &&\
	npm start

