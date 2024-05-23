// index.js

const express = require('express');
const { exec } = require('child_process');
const sqlite3 = require('sqlite3').verbose();
const app = express();
const port = 3033;
const path = require('path');
const fs = require('fs')
const cors = require('cors');
const jwt = require('jsonwebtoken');
const multer = require('multer');

require('dotenv').config()

// JWT secret key.
const secretKey = process.env.jwt_key

// Enable all CORS requests
const corsOptions = {
    origin: '*', // Adjust this to allow only specific domains
    methods: ['GET', 'POST', 'PUT', 'DELETE'], // Methods allowed for the CORS request
};

app.use(cors(corsOptions));
  
  



// Specify the env to use.
const python_env = process.env.python_env;
const windows = (process.env.windows == 'true'); // Switch to false if on linux

// Open the database
const db = new sqlite3.Database('./py/structures.db', sqlite3.OPEN_READWRITE | sqlite3.OPEN_CREATE, (err) => {

    if (err) {
        console.error(err.message)
    }

    console.log('Connected to the structures database.')
    initializeDatabase()
    initializeModels()

});

// Create our database if needed.
function initializeDatabase() {
    db.serialize(() => {

        db.run(`
        CREATE TABLE IF NOT EXISTS "dataset_block_counts" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_schematic_id INTEGER,
            block_id TEXT,
            count INTEGER,
            FOREIGN KEY (dataset_schematic_id) REFERENCES dataset_schematics (id),
            UNIQUE(dataset_schematic_id, block_id)
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS "model_block_counts" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_schematic_id INTEGER,
            block_id TEXT,
            count INTEGER,
            FOREIGN KEY (model_schematic_id) REFERENCES "model_schematics" (id),
            UNIQUE(model_schematic_id, block_id)
        ) 
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS "model_rankings" (
            model_id INTEGER,
            ranking_id INTEGER,
            FOREIGN KEY (model_id) REFERENCES models (id),
            FOREIGN KEY (ranking_id) REFERENCES "rankings" (id)
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS "rankings" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            minSimilarity INTEGER
        , "name" TEXT, "color" TEXT)
        `);

       
        db.run(`
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            display_name TEXT,
            epochs INTEGER DEFAULT NULL,
            learning_rate REAL DEFAULT NULL,
            batch_size INTEGER DEFAULT NULL,
            loss_method TEXT DEFAULT NULL,
            display_order INTEGER DEFAULT NULL,
            description TEXT DEFAULT NULL,
            hidden INTEGER DEFAULT 0,
            model_type INTEGER DEFAULT 0
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT
        , "entries" INTEGER, "size" INTEGER, hidden INTEGER DEFAULT 0)
        `);
        

        db.run(`
        CREATE TABLE IF NOT EXISTS dataset_schematics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filepath TEXT,
            dataset_id INTEGER,
            created_at TEXT, "favorite" INTEGER DEFAULT 0, category_id INTEGER DEFAULT (NULL) REFERENCES schematic_categories (id), 
            approved INTEGER DEFAULT 0, uploader_id INTEGER REFERENCES users (id) DEFAULT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS model_schematics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filepath TEXT,
            model_id INTEGER,
            created_at TEXT, favorite INTEGER DEFAULT 0, 
            category_id INTEGER DEFAULT (NULL) REFERENCES schematic_categories(id),
            FOREIGN KEY (model_id) REFERENCES models (id)
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS "divergence_score" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            w0_score REAL,
            whalf_score REAL,
            w1_score REAL,
            pattern_dim INTEGER,
            sample_size INTEGER,
            dataset_id INTEGER,
            model_id INTEGER,
            hidden INTEGER default 0,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id),
            FOREIGN KEY (model_id) REFERENCES models (id)
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS divergence_score_sample (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            divergence_score_id INTEGER,
            divergence_pattern_relationship_id INTEGER,
            FOREIGN KEY (divergence_pattern_relationship_id) REFERENCES divergence_pattern_relationship (id),
            FOREIGN KEY (divergence_score_id) REFERENCES divergence_score (id)
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS divergence_schematic_finished (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_dim INTEGER,
            model_schematic_id INTEGER DEFAULT NULL,
            dataset_schematic_id INTEGER DEFAULT NULL,
            FOREIGN KEY (model_schematic_id) REFERENCES model_schematics (id),
            FOREIGN KEY (dataset_schematic_id) REFERENCES dataset_schematics (id)
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS divergence_pattern_relationship (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            divergence_pattern_id INTEGER,
            model_schematic_id INTEGER DEFAULT NULL,
            dataset_schematic_id INTEGER DEFAULT NULL,
            count INTEGER,
            FOREIGN KEY (divergence_pattern_id) REFERENCES divergence_pattern (id),
            FOREIGN KEY (model_schematic_id) REFERENCES model_schematics (id),
            FOREIGN KEY (dataset_schematic_id) REFERENCES dataset_schematics (id)
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS divergence_pattern (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            npy_filepath TEXT,
            pattern_dim INTEGER
        )
        `);

        db.run(`
        CREATE TABLE IF NOT EXISTS "users" (
            id INT AUTO_INCREMENT PRIMARY KEY,
            token VARCHAR(255) DEFAULT NULL,
            email VARCHAR(255) DEFAULT NULL,
            password VARCHAR(255) DEFAULT NULL
        , type INT DEFAULT 0)
        `);

    });

    console.log('Database initialized with new tables.');

}

// Function to read models directory and insert models into the database
function initializeModels() {
    console.log('Checking for new models...')
    const modelsDir = path.join('py', 'models');
    fs.readdir(modelsDir, (err, files) => {
        if (err) {
            console.error('Error reading the models directory:', err);
            return;
        }

        files.forEach(file => {
            // Filter out files that do not end with '.pth.tar'
            if (file.endsWith('.pth.tar')) {
                // E-DCGAN vs RA-DCGAN
                var modelType = (file.startsWith('e_')) ? 1 : 0 
                //  ERA-DCGAN vs previous
                modelType = (file.match(/e_ra/)) ? 2 : modelType
                const name = file.replace('.pth.tar', '');
                const epochsMatch = file.match(/(\d+)epochs/);
                const epochs = epochsMatch ? parseInt(epochsMatch[1], 10) : null;
                const createdAt = new Date().toISOString().replace('T', ' ').replace('Z', '').split('.')[0];

                // Check if the model already exists
                db.get('SELECT name FROM models WHERE name = ?', [name], (err, row) => {
                    if (err) {
                        console.error('Error querying the database:', err);
                        return;
                    }
                    if (!row) {
                        // Model does not exist, insert new record
                        db.run('INSERT INTO models (name, epochs, created_at, model_type) VALUES (?, ?, ?, ?)', [name, epochs, createdAt, modelType], function(err) {
                            if (err) {
                                console.error('Error inserting model into database:', err);
                            } else {
                                console.log(`Model ${name} inserted into database.`);
                                // Retrieve the id of the newly inserted model
                                const modelId = this.lastID; // 'this' refers to the context of this callback
                                // Now, insert a new record into model_rankings with the modelId and a default ranking_id of 1
                                db.run('INSERT INTO model_rankings (model_id, ranking_id) VALUES (?, ?)', [modelId, 3], function(err) {
                                    if (err) {
                                        console.error('Error inserting model ranking into database:', err);
                                    } else {
                                        console.log(`Default model ranking for model ${modelId} inserted into database.`);
                                    }
                                });
                            }
                        });
                    } else {
                        console.log(`Model ${name} already exists in the database.`);
                    }                    
                });
            }
        });
    });
}

app.use(express.json()); // for parsing application/json

// Serve static files from the 'assets' folder
app.use('/assets', express.static('assets'));

// Helper function to interpolate between two colors
function interpolateColor(color1, color2, factor) {
    if (arguments.length < 3) { 
        factor = 0.5; 
    }
    let result = color1.slice();
    for (let i = 0; i < 3; i++) {
        result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
    }
    return result;
}

// Convert an RGB array to a hex string
function rgbToHex(rgb) {
    return "#" + rgb.map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
    }).join("");
}

// Helper function to determine color based on score
function getColor(score, minScore, maxScore) {
    const ratio = (score - minScore) / (maxScore - minScore);
    // Define start (red) and end (green) colors in RGB
    const startColor = [0, 100, 0]; // Green
    const endColor = [100, 0, 0]; // Red
    // Interpolate between the start and end colors based on score ratio
    const interpolatedColor = interpolateColor(startColor, endColor, ratio);
    return rgbToHex(interpolatedColor);
}

app.get('/heatmap-data', async (req, res) => {
    db.all(`
        SELECT ds.*, m.display_name AS model_name 
        FROM divergence_score ds 
        JOIN models m ON ds.model_id = m.id 
        WHERE ds.hidden=0
        ORDER BY ds.model_id, ds.pattern_dim
    `, [], async (err, rows) => {
        if (err) {
            console.error("Error fetching divergence scores:", err.message);
            return res.status(500).send({ error: 'Database error' });
        }

        const scores = rows.map(row => [row.w0_score, row.whalf_score, row.w1_score]).flat();
        const minScore = Math.min(...scores);
        const maxScore = Math.max(...scores);

        const heatmapModels = [];
        const modelScores = {}; // {model_id: {name, scores: [{pattern_dim, w0, whalf, w1}]}}

        // Define a consistent order for score types
        const scoreTypes = ['w0.0', 'w0.5', 'w1.0'];
        
        // Initialize heatmapTypes with a consistent order based on pattern_dim
        const heatmapTypes = new Set();
        rows.forEach(row => {
            scoreTypes.forEach(scoreType => {
                heatmapTypes.add(`${scoreType} ${row.pattern_dim}^3`);
            });
        });
        
        // Convert heatmapTypes Set to array to fix the order
        const heatmapTypesArray = Array.from(heatmapTypes);
        
        rows.forEach(row => {
            if (!modelScores[row.model_id]) {
                modelScores[row.model_id] = { name: row.model_name, scores: [] };
            }

            const scores = [
                { type: 'w0.0', score: row.w0_score, color: getColor(row.w0_score, minScore, maxScore) },
                { type: 'w0.5', score: row.whalf_score, color: getColor(row.whalf_score, minScore, maxScore) },
                { type: 'w1.0', score: row.w1_score, color: getColor(row.w1_score, minScore, maxScore) }
            ];

            modelScores[row.model_id].scores.push({
                pattern_dim: row.pattern_dim,
                scores // Keeping score types ordered as defined
            });
        });

        // Populate heatmapModels array from modelScores
        Object.keys(modelScores).forEach(modelId => {
            const model = modelScores[modelId];
            const orderedScores = model.scores.flatMap(scoreEntry => 
                scoreTypes.map(scoreType => {
                    const score = scoreEntry.scores.find(s => s.type === scoreType);
                    return { score: score.score, color: score.color };
                })
            );

            heatmapModels.push({
                name: model.name,
                scores: orderedScores
            });
        });

        res.send({ heatmapModels, heatmapTypes: heatmapTypesArray });
    });
});

app.get('/heatmap-data', async (req, res) => {
    db.all(`
        SELECT ds.*, m.name AS model_name 
        FROM divergence_score ds 
        JOIN models m ON ds.model_id = m.id 
        ORDER BY ds.model_id, ds.pattern_dim
    `, [], async (err, rows) => {
        if (err) {
            console.error("Error fetching divergence scores:", err.message);
            return res.status(500).send({ error: 'Database error' });
        }

        // Initialize necessary variables
        const modelScores = {}; // {model_id: {name, scores: [{pattern_dim, scores: [{type, score, color}]}]}}
        const patternDims = new Set();
        const heatmapTypes = new Set();

        // Populate patternDims and heatmapTypes
        rows.forEach(row => {
            patternDims.add(row.pattern_dim);
            ['w0.0', 'w0.5', 'w1.0'].forEach(type => {
                heatmapTypes.add(`${type} ${row.pattern_dim}x${row.pattern_dim}`);
            });
        });

        const heatmapTypesArray = Array.from(heatmapTypes);
        const scoreTypes = ['w0.0', 'w0.5', 'w1.0'];

        // Process each row to fill modelScores
        rows.forEach(row => {
            if (!modelScores[row.model_id]) {
                modelScores[row.model_id] = { name: row.model_name, scores: {} };
            }
            if (!modelScores[row.model_id].scores[row.pattern_dim]) {
                modelScores[row.model_id].scores[row.pattern_dim] = scoreTypes.map(type => ({
                    type,
                    score: 'N/A', // Default placeholder
                    color: '#000000' // Black
                }));
            }

            modelScores[row.model_id].scores[row.pattern_dim][0].score = row.w0_score.toString();
            modelScores[row.model_id].scores[row.pattern_dim][0].color = getColor(row.w0_score, minScore, maxScore);
            modelScores[row.model_id].scores[row.pattern_dim][1].score = row.whalf_score.toString();
            modelScores[row.model_id].scores[row.pattern_dim][1].color = getColor(row.whalf_score, minScore, maxScore);
            modelScores[row.model_id].scores[row.pattern_dim][2].score = row.w1_score.toString();
            modelScores[row.model_id].scores[row.pattern_dim][2].color = getColor(row.w1_score, minScore, maxScore);
        });

        // Convert modelScores to heatmapModels with proper padding
        const heatmapModels = Object.keys(modelScores).map(modelId => {
            const model = modelScores[modelId];
            const modelScoresFlattened = Object.keys(model.scores).flatMap(patternDim => 
                model.scores[patternDim]
            );

            return {
                name: model.name,
                scores: modelScoresFlattened
            };
        });

        res.send({ heatmapModels, heatmapTypes: heatmapTypesArray });
    });
});

const upload = multer({ dest: 'uploads/' }); // Temporarily store files in 'uploads' directory

app.post('/upload-schematic', upload.single('schemFile'), (req, res) => {
    const token = req.headers.authorization.split(' ')[1];
    if (!req.file || path.extname(req.file.originalname) !== '.schem') {
        return res.status(400).send({ error: 'File must be a .schem' });
    }

    try {
        const decoded = jwt.verify(token, secretKey);
        const userId = decoded.id;

        // Generate unique name and move file
        const uniqueName = `${Date.now()}-${req.file.originalname}`;
        const savePath = path.join(__dirname, 'py/schematics/dataset', uniqueName);
        const parsedPath = path.parse(savePath);
        const pathWithoutExtension = path.join(parsedPath.dir, parsedPath.name);
        fs.renameSync(req.file.path, savePath);

        // Run Python script
        const pythonScriptPath = path.join(__dirname, 'py', 'check_schem_dimensions.py');
        exec(`python ${pythonScriptPath} --schem_path "${pathWithoutExtension}"`, (error, stdout, stderr) => {
            if (error || stderr) {
                console.error(`Error running check_schem_dimensions.py: ${error || stderr}`);
                return res.status(500).send({ error: 'Failed to check schematic dimensions' });
            }

            // Check dimensions output
            if (stdout.trim() !== 'OK') {
                return res.status(400).send({ error: 'Schematic dimensions are out of bounds' });
            }

            // Save entry in database
            db.run("INSERT INTO dataset_schematics (name, filepath, dataset_id, created_at, uploader_id) VALUES (?, ?, 1, CURRENT_TIMESTAMP, ?)", [uniqueName, pathWithoutExtension, userId], function(err) {
                if (err) {
                    console.error('Database insert error:', err);
                    return res.status(500).send({ error: 'Failed to record schematic' });
                }

                res.send({ success: true, message: 'Schematic uploaded and validated' });
            });
        });

    } catch (error) {
        console.error(error);
        res.status(500).send({ success: false, message: 'Token check failed.' });
    }
 
});

// For verifying and decoding the user token
function verifyToken(token, id) {
    return new Promise((resolve, reject) => {
        jwt.verify(token, secretKey, (err, decoded) => {
            if (err) {
                reject(false);
            } else {
                resolve(decoded == id); // Decoded payload of the token
            }
        });
    });
}

// Extract a user id from token.
function parseToken(token) {
    return new Promise((resolve, reject) => {
        jwt.verify(token, secretKey, (err, decoded) => {
            if (err) {
                reject(err);
            } else {
                resolve(decoded); // Decoded payload of the token
            }
        });
    });
}

// Helper function to check if an email and password pair exists
function emailPasswordPairExists(email, hashedPassword) {
    return new Promise((resolve, reject) => {
        const query = "SELECT 1 FROM users WHERE email = ? AND password = ?";
        db.all(query, [email, hashedPassword], (err, rows) => {
            if (err) {
                reject(err);
            } else {
                resolve(rows.length > 0);
            }
        });
    });
}

async function emailPasswordPairExistsReturnUser(email, hashedPassword) {
    return new Promise((resolve, reject) => {
        const query = "SELECT * FROM users WHERE email = ? AND password = ?";
        db.all(query, [email, hashedPassword], (err, rows) => {
            if (err) {
                reject(err);
            } else {
                resolve(rows.length > 0 ? rows[0] : null); // Return user object if found
            }
        });
    });
}

// For creating a user token using JWT.
function generateToken(userId) {
    return jwt.sign({ id: userId }, secretKey, { expiresIn: '24h' });
}

function getUserById(userId) {
    return new Promise((resolve, reject) => {
        const query = "SELECT * FROM users WHERE id = ?";
        db.all(query, [userId], (err, rows) => {
            if (err) {
                console.error("Error fetching user:", err.message);
                reject(err);
            } else {
                if (rows.length > 0) {
                    resolve(rows[0]); // Assuming 'rows' is an array of results, return the first one.
                } else {
                    resolve(null); // No user found with the given ID.
                }
            }
        });
    });
}

function updateUserPassword(userId, newPasswordHash) {
    return new Promise((resolve, reject) => {
        const query = "UPDATE users SET password = ? WHERE id = ?";
        db.run(query, [newPasswordHash, userId], function(err) {
            if (err) {
                console.error("Error updating user password:", err.message);
                reject(err);
            } else {
                if (this.changes > 0) {
                    resolve(true); // Password update was successful.
                } else {
                    resolve(false); // No user found with the given ID, so no password was updated.
                }
            }
        });
    });
}

// Update a users password. given a valid JWT.4
app.post('/update-password', async (req, res) => {
    const token = req.headers.authorization.split(' ')[1];
    const { currentPassword, newPassword } = req.body;

    try {
        const decoded = jwt.verify(token, secretKey);
        const userId = decoded.id;
        // Assume a function exists to get the current hashed password for the user
        const currentUser = await getUserById(userId);
        
        const match = (currentUser.password === currentPassword) ? true : false
        if (!match) {
            return res.status(401).send({ success: false, message: 'Current password is incorrect.' });
        }

        await updateUserPassword(userId, newPassword);

        res.send({ success: true, message: 'Password updated successfully.' });
    } catch (error) {
        console.error(error);
        res.status(500).send({ success: false, message: 'Failed to update password.' });
    }
    
});

// Update an entry's favorite status.
app.post('/update-favorite', async (req, res) => {
    const { id, favorite, token } = req.body; // Assuming token is now part of the request

    if (!token || id === undefined || favorite === undefined || !(favorite == 0 || favorite == 1)) {
        return res.status(400).send({ error: "Missing token, id, or valid favorite in request" });
    }

    try {
        const isValidToken = await verifyToken(token, id);
        if (!isValidToken) {
            return res.status(401).send({ error: "Invalid or missing authentication token" });
        }

        const query = "UPDATE model_schematics SET favorite = ? WHERE id = ?";
        db.run(query, [favorite, id], function(err) {
            if (err) {
                console.error("Error updating favorite status:", err.message);
                return res.status(500).send({ error: err.message });
            }

            if (this.changes === 0) {
                return res.status(404).send({ error: "Schematic not found" });
            }

            res.send({ success: true, message: "Favorite status updated", id: id, favorite: favorite });
        });
    } catch (err) {
        console.error("Database error:", err.message);
        res.status(500).send({ error: "Database error" });
    }
});

// Refresh a users token and return their info in if they pass the match.
app.post('/login', async (req, res) => {
    const { email, password } = req.body; // 'password' should already be hashed

    if (!email || !password) {
        return res.status(400).send({ success: false, error: "Missing email or password" });
    }

    try {
        const user = await emailPasswordPairExistsReturnUser(email, password);
        if (user) {

            const token = generateToken(user.id);
            user.token = token

            res.send({ success: true, user, message: "Login successful" });
        } else {
            // Authentication failed
            res.status(401).send({ success: false, error: "Authentication failed" });
        }
    } catch (err) {
        console.error("Database error:", err.message);
        res.status(500).send({ success: false, error: "Database error" });
    }
});

// Fetch by model category with pagination
app.get('/model-schematics', (req, res) => {

    const { page = 1, limit = 3, favoritesOnly = false } = req.query;
    let { model } = req.query;
    const offset = (page - 1) * limit;
    const favoritesCondition = favoritesOnly === 'true' ? 'AND model_schematics.favorite = 1' : ''; // Adjust based on actual type of favoritesOnly

    // Modified query to include rankings
    const modelsSql = `
        SELECT models.display_name, models.name, rankings.id AS ranking_id, rankings.minSimilarity
        FROM models
        JOIN model_rankings ON models.id = model_rankings.model_id
        JOIN rankings ON model_rankings.ranking_id = rankings.id
        WHERE models.hidden=0
        ORDER BY rankings.minSimilarity ASC, models.display_order ASC
    `;

    const rankingsSql = `
        SELECT * FROM rankings ORDER BY minSimilarity DESC
    `;

    db.all(modelsSql, [], (err, models) => {
        if (err) {
            console.log('modelsSql err:',err)
            return res.status(500).send({ error: err.message });
        }

        if (models.length === 0) {
            return res.send({ models: [], data: [], more: false });
        }

        // If our client provided model is undefined, use the first one here.
        if (!model && models.length > 0) {
            model = models[0].display_name;
        }

         // Prepare models grouped by rankings
        const maxRankingId = models.reduce((max, {ranking_id}) => Math.max(max, ranking_id), 0);
        const groupedModels = Array.from({ length: maxRankingId }).map(() => []);

        models.forEach(model_inspect => {
            // Adjusted to zero-based index for array
            groupedModels[model_inspect.ranking_id - 1].push(model_inspect.display_name);
        });

        // remove any empty ranking slots if there are gaps in ranking_id, reverse so top rank is first
        const filledRankedModels = groupedModels.filter(rankGroup => rankGroup.length > 0).reverse();

        db.all(rankingsSql, [], (err, rankings) => {
            if (err) {
                console.log('rankingsSql err:',err)
                return res.status(500).send({ error: err.message });
            }

            // After fetching models and before executing the main query for schematics
            const averageModelCountsSql = `
                SELECT bc.block_id, AVG(bc.count) AS average_count
                FROM dataset_block_counts AS bc
                JOIN dataset_schematics AS ms ON bc.dataset_schematic_id = ms.id
                GROUP BY bc.block_id
            `;

            // Declare a variable to hold the average counts result
            let averageModelCounts = [];

            const countSql = `
                SELECT COUNT(*) AS total
                FROM model_schematics
                JOIN models ON model_schematics.model_id = models.id
                WHERE models.display_name = ? and models.hidden=0 ${favoritesCondition}
            `;

            db.get(countSql, [model], (err, { total }) => {
                if (err) {
                    console.log('countSql err:',countSql)
                    return res.status(500).send({ error: err.message });
                }

                // Fetch the average block counts for the specified model
                db.all(averageModelCountsSql, [], (err, averages) => {
                    if (err) {
                        console.log('averageModelCountsSql err:', err);
                        return res.status(500).send({ error: err.message });
                    }
                    averageModelCounts = averages; // Store the averages for inclusion in the response

                    const sql = `
                        SELECT model_schematics.*
                        FROM model_schematics
                        JOIN models ON model_schematics.model_id = models.id
                        WHERE models.display_name = ? and models.hidden=0 ${favoritesCondition}
                        ORDER BY model_schematics.id DESC
                        LIMIT ? OFFSET ?
                    `;
                    
                    db.all(sql, [model, limit, offset], (err, model_schematics) => {
                        
                        if (err) {
                            console.log('sql err:', err);
                            return res.status(500).send({ error: err.message });
                        }
                    
                        const promises = model_schematics.map(schematic => {
                            return new Promise((resolve, reject) => {
                                // Check if the app is running on a non-Windows system and adjust the file path accordingly
                                //console.log('windows:',typeof(windows))
                                const adjustedFilePath = !windows ? schematic.filepath.replace(/\\/g, '/') : schematic.filepath;
                                const filePath = path.join(adjustedFilePath); // Use the adjusted file path here
                    
                                fs.readFile(filePath, {encoding: 'base64'}, (err, data) => {
                                    if (err) {
                                        reject(err);
                                    } else {

                                        const blockCountsSql = `
                                            SELECT block_id, count
                                            FROM model_block_counts
                                            WHERE model_schematic_id = ?
                                            ORDER BY block_id DESC
                                        `;
                    
                                        db.all(blockCountsSql, [schematic.id], (err, blockCounts) => {
                                            if (err) {
                                                reject(err);
                                            } else {
                                                // Combine the schematic data with the base64 and block counts
                                                resolve({...schematic, base64: data, blockCounts});
                                            }
                                        });
                                    }
                                });
                            });
                        });
                    
                        Promise.all(promises)
                            .then(results => {
                                const more = total > offset + results.length; // Check if there are more entries
                    
                                res.send({
                                    models: filledRankedModels,
                                    data: results,
                                    more, // Indicates if there are more entries to load
                                    total,
                                    rankings,
                                    averageModelCounts // TODO: we need to make this a thing
                                });
                            })
                            .catch(error => {
                                console.log('promise err:', error);
                                res.status(500).send({ error: error.message });
                            });
                    });

                });
            
            });

        });

        
    });
});

function extractRGANumber(str, modelType) {

    // Skip if not the RGA generator.
    if (modelType != 0) {
        return 0
    }

    // Regular expression to find 'rga' followed by one or more digits
    const regex = /rga(\d+)/;
    
    // Use the regex to search the string
    const match = str.match(regex);
    
    // If a match is found, return the number part, otherwise return null
    if (match) {
      // Convert the matched string to a number
      return parseInt(match[1], 10);
    } else {
        // Default RGA size
      return 2;
    }
}

// Generate structure endpoint updated to check model name and fetch schematic details
app.get('/generate-structure', async (req, res) => {
    const { modelName, token } = req.query;

    try {
        let verified = await parseToken(token)

        console.log('/generate-structure')
        console.log('verified:',verified)
        
        db.get("SELECT id, model_type FROM models WHERE name = ?", [modelName], (err, modelRow) => {
            if (err) {
                return res.status(500).send({ error: err.message });
            }
            if (!modelRow) {
                return res.status(404).send({ error: "Model not found" });
            }

            const modelType = modelRow["model_type"]
            console.log(modelType)

            var execline = windows ? `py\\${python_env}\\Scripts\\python py\\generate_structure.py ${modelName} ${modelType} ${extractRGANumber(modelName, modelType)}` : `./py/${python_env}/bin/python ./py/generate_structure.py ${modelName} ${modelType} ${extractRGANumber(modelName, modelType)}`;

            exec(execline, (error, stdout, stderr) => {

                if (error) {
                    console.error(`exec error: ${error}`);
                    return res.status(500).send({ error: `Execution error: ${error.message}` });
                }

                if (stderr) {
                    console.error(`stderr: ${stderr}`);
                    return res.status(500).send({ error: stderr });
                }

                const outputLines = stdout.trim().split('\n');
                const filename = outputLines[0].trim(); // Trim filename to remove any leading/trailing whitespace
                const base64_schem_content = outputLines[1];

                console.log('created',filename)

                // Query the model_schematics table using the filename
                db.get("SELECT * FROM model_schematics WHERE name = ?", [filename], (err, schematicRow) => {
                    
                    if (err) {
                        return res.status(500).send({ error: err.message });
                    }
                    if (!schematicRow) {
                        console.log('Schematic not found in database:', filename);
                        return res.status(404).send({ error: "Schematic not found" });
                    }

                    // Prepare the data object
                    const data = {
                        base64: base64_schem_content,
                        ...schematicRow
                    };

                    res.send({ data });

                });
            });
        });

    } catch (err) {

        console.log('/generate-structure error:')
        console.log(err)

    }
});

// Compute and update block distributions rankings.
app.post('/update-block-distributions', (req, res) => {

    // Run our block_distribution.py to scan our model_schematics directory and compute distributions.
    var execline = windows ? `py\\${python_env}\\Scripts\\python py\\model_block_distributions.py` : `./py/${python_env}/bin/python ./py/model_block_distributions.py`;

    exec(execline, (error, stdout, stderr) => {

        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send({ error: `Execution error: ${error.message}` });
        }

        if (stderr) {
            console.error(`stderr: ${stderr}`);
            return res.status(500).send({ error: stderr });
        }

        var datasetexecline = windows ? `py\\${python_env}\\Scripts\\python py\\dataset_block_distributions.py` : `./py/${python_env}/bin/python ./py/dataset_block_distributions.py`;

        const model_stdout = stdout

        exec(datasetexecline, (error, stdout, stderr) => {

            if (error) {
                console.error(`exec error: ${error}`);
                return res.status(500).send({ error: `Execution error: ${error.message}` });
            }

            if (stderr) {
                console.error(`stderr: ${stderr}`);
                return res.status(500).send({ error: stderr });
            }

            const final = { success: true, model_stdout, dataset_stdout:stdout }

            console.log(final)
            res.send(final);

        });

        

    });

});

// Run our matrix pattern KL divergence calculations.
app.post('/update-pattern-distributions', (req, res) => {

    const { sample_size, pattern_dim } = req.body;

    // Run our block_distribution.py to scan our model_schematics directory and compute distributions.
    var execline = windows ? `py\\${python_env}\\Scripts\\python py\\pattern_distributions.py --pattern_dim ${pattern_dim} --sample_size ${sample_size}` : `./py/${python_env}/bin/python ./py/pattern_distributions.py --pattern_dim ${pattern_dim} --sample_size ${sample_size}`;

    exec(execline, (error, stdout, stderr) => {

        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send({ error: `Execution error: ${error.message}` });
        }

        if (stderr) {
            console.error(`stderr: ${stderr}`);
            return res.status(500).send({ error: stderr });
        }

        const final = { success: true, stdout }

        res.send(final)

    });

});

app.get('/dataset-schematics', (req, res) => {
    const { page = 1, limit = 3, favoritesOnly = false } = req.query;
    let { dataset } = req.query;
    const offset = (page - 1) * limit;
    const favoritesCondition = favoritesOnly === 'true' ? 'AND dataset_schematics.favorite = 1' : '';

    const datasetsSql = `
        SELECT name FROM datasets WHERE hidden=0
    `;

    db.all(datasetsSql, [], (err, datasets) => {
        if (err) {
            console.log('datasetsSql err:', err)
            return res.status(500).send({ error: err.message });
        }

        if (datasets.length === 0) {
            return res.send({ datasets: [], data: [], more: false });
        }

        // If our client provided dataset is undefined, use the first one here.
        if (!dataset && datasets.length > 0) {
            dataset = datasets[0].name;
        }

        // After fetching models and before executing the main query for schematics
        const averageDatasetCountsSql = `
            SELECT bc.block_id, AVG(bc.count) AS average_count
            FROM dataset_block_counts AS bc
            JOIN dataset_schematics AS ms ON bc.dataset_schematic_id = ms.id
            GROUP BY bc.block_id
        `;

        // Declare a variable to hold the average counts result
        let averageDatasetCounts = [];

        const countSql = `
            SELECT COUNT(*) AS total
            FROM dataset_schematics
            JOIN datasets ON dataset_schematics.dataset_id = datasets.id
            WHERE datasets.name = ? ${favoritesCondition}
        `;

        db.get(countSql, [dataset], (err, { total }) => {
            if (err) {
                console.log('countSql err:', err)
                return res.status(500).send({ error: err.message });
            }

            // Fetch the average block counts
            db.all(averageDatasetCountsSql, [], (err, averages) => {
                if (err) {
                    console.log('averageDatasetCountsSql err:', err);
                    return res.status(500).send({ error: err.message });
                }
                averageDatasetCounts = averages; // Store the averages for inclusion in the response


                const sql = `
                    SELECT dataset_schematics.*
                    FROM dataset_schematics
                    JOIN datasets ON dataset_schematics.dataset_id = datasets.id
                    WHERE datasets.name = ? AND approved=1 ${favoritesCondition}
                    ORDER BY dataset_schematics.id ASC
                    LIMIT ? OFFSET ?
                `;

                db.all(sql, [dataset, limit, offset], (err, dataset_schematics) => {
                    if (err) {
                        console.log('sql err:', err);
                        return res.status(500).send({ error: err.message });
                    }

                    const more = total > offset + dataset_schematics.length; // Check if there are more entries

                    const promises = dataset_schematics.map(schematic => {
                        return new Promise((resolve, reject) => {
                            // Check if the app is running on a non-Windows system and adjust the file path accordingly
                            //console.log('windows:',typeof(windows))
                            const adjustedFilePath = !windows ? schematic.filepath.replace(/\\/g, '/') : schematic.filepath;
                            const filePath = path.join(adjustedFilePath) + ".schem"; // Use the adjusted file path here
                
                            fs.readFile(filePath, {encoding: 'base64'}, (err, data) => {
                                if (err) {
                                    reject(err);
                                } else {

                                    const blockCountsSql = `
                                        SELECT block_id, count
                                        FROM dataset_block_counts
                                        WHERE dataset_schematic_id = ?
                                        ORDER BY block_id DESC
                                    `;
                
                                    db.all(blockCountsSql, [schematic.id], (err, blockCounts) => {
                                        if (err) {
                                            reject(err);
                                        } else {
                                            // Combine the schematic data with the base64 and block counts
                                            resolve({...schematic, base64: data, blockCounts});
                                        }
                                    });
                                }
                            });
                        });
                    });
                
                    Promise.all(promises)
                        .then(results => {

                            res.send({
                                datasets: datasets.map(d => d.name), // Return dataset names for consistency with model endpoint
                                data: results,
                                more, // Indicates if there are more entries to load
                                total,
                                averageDatasetCounts
                            });
                        })
                        .catch(error => {
                            console.log('promise err:', error);
                            res.status(500).send({ error: error.message });
                        });

                });

                

            });

        });

    });

});

app.post('/update-dataset-favorite', (req, res) => {
    const { id, favorite } = req.body;

    if (id === undefined || favorite === undefined || !(favorite == 0 || favorite == 1)) {
        return res.status(400).send({ error: "Missing id or valid favorite in request" });
    }

    const query = "UPDATE dataset_schematics SET favorite = ? WHERE id = ?";

    db.run(query, [favorite, id], function(err) {
        if (err) {
            console.error("Error updating favorite status:", err.message);
            return res.status(500).send({ error: err.message });
        }

        if (this.changes === 0) {
            return res.status(404).send({ error: "Schematic not found" });
        }

        res.send({ success: true, message: "Favorite status updated", id: id, favorite: favorite });
    });
});


app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
