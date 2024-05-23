// API.js

import * as Crypto from 'expo-crypto'

//const API_URL = "http://localhost:3033"
const API_URL = "https://api-minecraftgan.adambullard.com"

/**
 * Fetches heatmap data from the server.
 * @returns {Promise<{heatmapModels: Array, heatmapTypes: Array}>} 
 * A promise that resolves with heatmap models and types.
 */
export const fetchHeatmapData = async () => {

    try {
        const url = new URL(API_URL + "/heatmap-data"); 

        const response = await fetch(url);
        if (!response.ok) {
            console.log(response);
            throw new Error('Network response was not ok');
        }

        const { heatmapModels, heatmapTypes } = await response.json();

        console.log(heatmapModels, heatmapTypes);
        return { heatmapModelsNew:heatmapModels, heatmapTypesNew:heatmapTypes };

    } catch (error) {
        console.error('Failed to fetch heatmap data:', error);
        throw error;
    }
};

/**
 * Update a user's password.
 * @param {string} token - JSON Web Token
 * @param {string} currentPassword - encrypted
 * @param {string} newPassword - encrypted
 * @returns {Promise<object>} A promise that resolves with an object containing:
 * - .success : 0/1
 */
export const updatePassword = async (token, currentPassword, newPassword) => {
    try {
        const encryptedCurrentPassword = await Crypto.digestStringAsync(
            Crypto.CryptoDigestAlgorithm.SHA256,
            currentPassword
        );
        const encryptedNewPassword = await Crypto.digestStringAsync(
            Crypto.CryptoDigestAlgorithm.SHA256,
            newPassword
        );

        const response = await fetch(`${API_URL}/update-password`, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`, // Use token for authentication
            },
            body: JSON.stringify({
                currentPassword: encryptedCurrentPassword,
                newPassword: encryptedNewPassword,
            }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        return result.success;
    } catch (error) {
        console.error('Failed to update password:', error);
        return false;
    }
};

/**
 * Fetches dataset schematic generations from the server.
 * @param {string} email - email address
 * @param {string} password - encrypted
 * @returns {Promise<object>} A promise that resolves with an object containing:
 * - .success : 0/1
 * - .user : user data object if success=1
 */
export async function loginCheck(email, password) {

    var ret = {success:false}
  
    // Encrypt Password.
    var pw = await Crypto.digestStringAsync(
      Crypto.CryptoDigestAlgorithm.SHA256,
      password
    )
  
    var arr = {email, password:pw}
  
    console.log('Checking login credentials...')
    console.log('Login arr:',arr)
    const res = await fetch(API_URL + '/login', {
      method:'POST',
      body: JSON.stringify(arr),
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      }
    })
  
    const payload = await res.json()
    console.log('Returning payload:',payload)
    if (payload.success == true) {
      console.log('Login successful!')
      ret = payload
    } else {
      console.log('Login failed!')
    }
  
    return ret
  
}

/**
 * Fetches dataset schematic generations from the server.
 * @param {string} targetDataset - The dataset name to fetch schematics for.
 * @param {number} page - The page number for pagination.
 * @param {number} limit - The number of items per page.
 * @param {boolean} favoritesOnly - Limit to favorites only or not.
 * @returns {Promise<[Array, Array, boolean]>} A promise that resolves with the new generations, datasets, and more flag.
 */
export const fetchDatasetModelGenerations = async (targetDataset, page=1, limit=3, favoritesOnly=false) => {
    const target = "/dataset-schematics";

    try {
        const url = new URL(API_URL + target);
        if (targetDataset) url.searchParams.append('dataset', targetDataset);
        url.searchParams.append('page', page);
        url.searchParams.append('limit', limit);
        url.searchParams.append('favoritesOnly', favoritesOnly);

        const response = await fetch(url);
        if (!response.ok) {
            console.log(response)
            throw new Error('Network response was not ok');
        }

        const { data, datasets, more, total, averageDatasetCounts } = await response.json();

        console.log('here', data, datasets, more, total, averageDatasetCounts)
        return [data, datasets, more, total, averageDatasetCounts];

    } catch (error) {
        console.error('Failed to fetch dataset generations:', error);
        throw error;
    }
};

/**
 * Updates the favorite status of a dataset schematic.
 * @param {number} id - The ID of the dataset schematic to update.
 * @param {boolean} favorite - The new favorite status.
 * @returns {Promise<void>} A promise that resolves when the operation is complete.
 */
export const updateDatasetFavoriteStatus = async (id, favorite) => {
    const target = "/update-dataset-favorite";

    try {
        const url = new URL(API_URL + target);

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ id, favorite }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        console.log('Update dataset favorite result:', result);

    } catch (error) {
        console.error('Failed to update dataset favorite status:', error);
        throw error;
    }
};

/**
 * Fetches schematic generations from the server.
 * @param {string} targetModel - The model name to fetch schematics for.
 * @param {number} page - The page number for pagination.
 * @param {number} limit - The number of items per page.
 * @param {boolean} favoritesOnly - Limit to favorites only or not.
 * @returns {Promise<[Array, Array, boolean]>} A promise that resolves with the new generations, models, and more flag.
 */
export const fetchModelGenerations = async (targetModel, page=1, limit=3, favoritesOnly=false) => {

    const target = "/model-schematics"

    try {

        const url = new URL(API_URL + target);
        if (targetModel) url.searchParams.append('model', targetModel);
        url.searchParams.append('page', page);
        url.searchParams.append('limit', limit);
        url.searchParams.append('favoritesOnly', favoritesOnly)

        const response = await fetch(url);
        if (!response.ok) {
            console.log(response)
            throw new Error('Network response was not ok');
        }

        const { data, models, more, total, rankings, averageModelCounts } = await response.json();
        
        console.log(data, models, more, total, rankings, averageModelCounts)

        return [data, models, more, total, rankings, averageModelCounts];

    } catch (error) {

        console.error('Failed to fetch generations:', error);
        throw error; 
        
    }

};

/**
 * Fetches a schematic generation for a specific model from the server.
 * @param {string} modelName - The name of the model to generate a schematic for.
 * @returns {Promise<string>} A promise that resolves with the generated schematic data.
 */
export const fetchNewModelGeneration = async (modelName, token) => {

    const target = "/generate-structure";

    try {
        const url = new URL(API_URL + target);
        if (modelName) url.searchParams.append('modelName', modelName);
        if (token) url.searchParams.append('token', token)

        const response = await fetch(url);
        if (!response.ok) {
            console.log('response:',response)
            throw new Error('Network response was not ok');
        }

        const { data } = await response.json();
        
        return data;

    } catch (error) {
        console.error('Failed to fetch schematic generation:', error);
        throw error;
    }
};

/**
 * Updates the favorite status of a schematic.
 * @param {number} id - The ID of the schematic to update.
 * @param {boolean} favorite - The new favorite status.
 * @returns {Promise<void>} A promise that resolves when the operation is complete.
 */
export const updateFavoriteStatus = async (id, favorite) => {
    const target = "/update-favorite";

    try {
        const url = new URL(API_URL + target);

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ id, favorite }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        console.log('Update result:', result);

    } catch (error) {
        console.error('Failed to update favorite status:', error);
        throw error;
    }
};


/**
 * Updates the favorite status of a schematic
 * @returns {Promise<string>} A promise that resolves when the operation is complete.
 */
export const updateBlockDistributions = async () => {
    const target = "/update-block-distributions";

    try {
        const url = new URL(API_URL + target);

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        console.log('updateBlockDistributions result:', result);

    } catch (error) {
        console.error('Failed to update favorite status:', error);
        throw error;
    }
};


/**
 * Updates the favorite status of a schematic
 * @param {number} sample_size - How many samples from each dataset and model to compare.
 * @param {number} pattern_dim - How large our pattern matrices should be. 
 * @returns {Promise<string>} A promise that resolves when the operation is complete.
*/
export const updatePatternDistributions = async (sample_size, pattern_dim) => {
    const target = "/update-pattern-distributions";

    try {

        const url = new URL(API_URL + target);

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pattern_dim, sample_size }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        console.log('updatePatternDistributions result:', result);

    } catch (error) {

        console.error('Failed to update favorite status:', error);
        throw error;

    }
};

/**
 * Uploads a single schematic file to the server.
 * @param {FormData} formData - The form data with the file to upload.
 * @returns {Promise<void>} A promise that resolves when the upload is complete.
 */
export const uploadSingleSchematic = async (formData, token) => {

    const target = "/upload-schematic"; 
  
    try {

      const url = new URL(API_URL + target);
      const response = await fetch(url, {
        method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'multipart/form-data',
                'Authorization': `Bearer ${token}`, // Use token for authentication
            },
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
  
      const result = await response.json();
      console.log('Upload single schematic result:', result);

      return result

    } catch (error) {

      console.error('Failed to upload single schematic:', error);
      throw error;

    }

};
  