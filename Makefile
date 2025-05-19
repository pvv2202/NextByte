backend:
	cd server && python3 -m venv venv && \
	source venv/bin/activate && \
	export FLASK_APP=server.py && \
	export FLASK_ENV=development && \
	flask run

react-app:
	cd frontend &&\
	npm start

