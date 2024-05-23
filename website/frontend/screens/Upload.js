// Upload.js

import React, { useState, useEffect, useCallback } from 'react';
import { Image, StatusBar, Text, View, Pressable, ActivityIndicator, ScrollView, Dimensions, Animated, TextInput, Switch  } from 'react-native';
import SchematicViewer from "../react-schematicwebviewer/dist";
import { Buffer } from 'buffer';
import styles from '../styles/Upload';
import { darkColors } from '../styles/Colors'
import { uploadSingleSchematic } from '../API'; 
import * as SplashScreen from 'expo-splash-screen';
import { AntDesign } from '@expo/vector-icons'
import Header from './shared/Header'
import Dropzone from 'react-dropzone'

SplashScreen.preventAutoHideAsync();

const screenWidth = Dimensions.get('window').width;

export default function Upload({ navigation, route }) {

  const routeName = (route.params != undefined) ? route.params.name : '/upload' 

  const [isMainLoading, setIsMainLoading] = useState(true)
  const [files, setFiles] = useState([])

  useEffect(() => {

    setIsMainLoading(false)

  }, [])

  // This should be given the incoming files info from the system.
  const processReceivedFiles = async (acceptedFiles) => {

    // Map incoming files to an initial state with a status of 0 (uploading)
    const initialFiles = acceptedFiles.map(file => ({ ...file, status: 0 }));
    setFiles(initialFiles);

    // Process each file individually
    for (const [index, file] of initialFiles.entries()) {
      try {
        const formData = new FormData();
        formData.append('file', file); // Assuming 'file' is an object suitable for FormData
        const result = await uploadSingleSchematic(formData); // Assuming this function exists and works correctly

        // Update the file status to appropriate status.
        setFiles(currentFiles => currentFiles.map((f, i) => i === index ? { ...f, status: result.status } : f));
      
      } catch (error) {
      
        console.error("Error uploading file:", error);
        // Update the file status to 2 (error) if upload fails
        setFiles(currentFiles => currentFiles.map((f, i) => i === index ? { ...f, status: 2 } : f));
      
      }

    }

  };
  
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Header routeName={routeName} navigation={navigation} />
      {isMainLoading ? (<ScrollView contentContainerStyle={[styles.middleSection,
        {
          justifyContent:'center',
          alignItems:'center'
        }
      ]}>

        <ActivityIndicator />

      </ScrollView>) : (<View style={styles.middleSection}>
        <View style={styles.uploadSection}>
          <Text style={styles.uploadTitle}>Upload Candidate Structures</Text>
          <View style={styles.uploadStipulations}>
              <Text style={styles.uploadStipulationTitle}>Requirements:</Text>
              <Text style={styles.uploadStipulationText}>1. .schem filetype from Minecraft 1.20.2 or below</Text>
              <Text style={styles.uploadStipulationText}>2. Less than 32 in length in every dimension</Text>
          </View>
          <Text style={styles.uploadTitleFiles}>Files</Text>
          <View style={styles.fileUploadSection}>
            {files.map(() => {

              return (<View style={styles.fileRow}>
                <View style={styles.fileNameWrapper}>
                  <Text style={styles.fileName}>{file.name}</Text>
                </View>
                <View style={styles.fileStatus}>
                  {file.status == 0 && (<ActivityIndicator />)}
                  {file.status == 1 && (<AntDesign 
                    name={''} 
                    size={20} 
                    color={darkColors.green} 
                  />)}
                  {file.status == 2 && (<AntDesign 
                    name={''} 
                    size={20} 
                    color={darkColors.red} 
                  />)}
                </View>
              </View>)
            })}
          </View>
          <View style={styles.dropzoneWrapper}>
            <Dropzone onDrop={acceptedFiles => processReceivedFiles(acceptedFiles)}>
              {({getRootProps, getInputProps}) => (
                <section>
                  <div {...getRootProps()}>
                    <input {...getInputProps()} />
                    <Text style={styles.uploadButtonText}>Drop .schems here</Text>
                  </div>
                </section>
              )}
            </Dropzone>
          </View>
        </View>
      </View>)}
      <StatusBar style="auto" />
    </ScrollView>
  );
}
