# Development Demo Live

Visit at: [minecraftgan.adambullard.com](https://minecraftgan.adambullard.com)

# Frontend 

Prereqs:
1. Install nodejs

Setup (from top level directory):
1. Run `npx npm install` inside `frontend/`
2. Start the backend API to enable full functionality.
3. Specify the API URL in `frontend/API.js`
4. From `frontend/` run `npx expo start --web`

# Backend

Prereqs:
1. Install nodejs
2. Install python

Setup (from top level directory):
1. Create a venv with `python -m venv backend/py/<env name>` 
2. Activate your venv with `backend/py/<env name>/Scripts/activate` on Windows or `backend/py/<env name>/bin/activate` in a bash environment
3. Install reqs to your venv with `pip install -r backend/py/requirements.txt`
4. Specify this env name in the `python_env` variable at the top of `backend/index.js` 
5. Place a model in `backend/py/models` and specify the checkpoint name in `backend/index.js`
6. Run `npx npm install` inside `backend/`
6. Start the server with `node backend/index.js`
