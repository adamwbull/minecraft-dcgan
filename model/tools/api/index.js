import express from 'express';
import mysql from 'mysql';
import cors from 'cors';
import dotenv from 'dotenv';
dotenv.config();

var app = express();
var router = express.Router();

// Enable CORS.
app.use(cors());

// Built-in Express body parser.
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));

// Enable routing.
app.use('/', router);

const config = {
    databaseOptions: {
      host : "localhost",
      port : process.env.MYSQL_PORT,
      user : process.env.MYSQL_USER,
      password : process.env.MYSQL_PASSWORD,
      database : process.env.MYSQL_DATABASE,
      multipleStatements: true,
      charset : 'utf8mb4',
      connectionLimit: 400
    }
};

var pool = mysql.createPool(config.databaseOptions);

// Use this function to make cleaner SQL changes.
async function execute_async (q, a) {

    return await new Promise(function(resolve,reject){
      pool.query(
        q,
        a,
        function (error, results, fields) {
          if (error) {
            reject(error);
          } else {
            resolve(results)
          }
        })
    })

}

// == Security == //

function verifyToken(req, res, next) {

    const bearerHeader = req.headers['authorization'];

    if (typeof bearerHeader !== 'undefined') {
        const bearerToken = bearerHeader.split(' ')[1];
        if (bearerToken === process.env.API_TOKEN) {
            next();
        } else {
            res.sendStatus(403);
        }
    } else {
        res.sendStatus(403);
    }

}

// == Endpoints == //

router.get('/entries', verifyToken, async (req, res) => {
    let { page = 1, pageSize = 64, random = false } = req.query;

    page = parseInt(page);
    pageSize = parseInt(pageSize);

    let query;
    if (random) {
        query = "SELECT * FROM entries ORDER BY shuffle_order LIMIT ?, ?";
    } else {
        query = "SELECT * FROM entries LIMIT ?, ?";
    }
    
    try {
        const entries = await execute_async(query, [(page - 1) * pageSize, pageSize]);
        res.status(200).send(entries);
    } catch (error) {
        res.status(500).send({ success: false, message: error.message });
    }
});


// Insert an entry, or update if non null/empty schematic_name has been seen already.
router.post('/entry', verifyToken, async (req, res) => {

    const { data, dim_x, dim_y, dim_z, source, schematic_name } = req.body;

    // If schematic_name is provided, check for an existing entry
    if (schematic_name && schematic_name.trim() !== '') {
        const checkQuery = "SELECT * FROM entries WHERE schematic_name=?";
        try {
            const entries = await execute_async(checkQuery, [schematic_name]);
            if (entries.length > 0) {
                // Entry exists, perform an update
                const updateQuery = "UPDATE entries SET data=?, dim_x=?, dim_y=?, dim_z=?, source=? WHERE schematic_name=?";
                await execute_async(updateQuery, [data, dim_x, dim_y, dim_z, source, schematic_name]);
                res.status(200).send({ success: true, operation: 'update' });
                return;
            }
        } catch (error) {
            res.status(500).send({ success: false, message: error.message });
            return;
        }
    }

    // If no matching entry was found or schematic_name was not provided, perform an insert
    try {
        const insertQuery = "INSERT INTO entries (data, dim_x, dim_y, dim_z, source, schematic_name) VALUES (?, ?, ?, ?, ?, ?, ?)";
        await execute_async(insertQuery, [data, dim_x, dim_y, dim_z, source, schematic_name]);
        res.status(201).send({ success: true, operation: 'insert' });
    } catch (error) {
        res.status(500).send({ success: false, message: error.message });
    }

});

// == Startup == //
async function createTableIfNotExists() {

    const query = `
    CREATE TABLE IF NOT EXISTS entries (
        id INT PRIMARY KEY AUTO_INCREMENT,
        shuffle_order INT NOT NULL,
        data MEDIUMBLOB NOT NULL,
        dim_x INT NOT NULL,
        dim_y INT NOT NULL,
        dim_z INT NOT NULL,
        source TEXT NOT NULL,
        schematic_name TEXT
    );
    `;

    try {
        await execute_async(query);
        console.log("Table 'entries' checked/created successfully.")
    } catch (error) {
        console.error("Error creating table: ", error);
    }

}


createTableIfNotExists();
const server = app.listen(process.env.API_PORT, function () {

    var port = server.address().port;
    console.log("\nApp live on " + port);

});