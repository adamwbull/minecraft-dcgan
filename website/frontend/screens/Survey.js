// Survey.js

import React, { useState, useEffect, useCallback } from 'react';
import { Image, StatusBar, Text, View, Pressable, ActivityIndicator, ScrollView, Dimensions, Animated, TextInput, Switch  } from 'react-native';
import SchematicViewer from "../react-schematicwebviewer/dist";
import { Buffer } from 'buffer';
import styles from '../styles/Rankings';
import { darkColors } from '../styles/Colors'
import { fetchDatasetModelGenerations, updateDatasetFavoriteStatus } from '../API'; 
import * as SplashScreen from 'expo-splash-screen';
import { AntDesign } from '@expo/vector-icons'
import Header from './shared/Header'
import { fetchHeatmapData } from '../API.js'

SplashScreen.preventAutoHideAsync();

const screenWidth = Dimensions.get('window').width;

export default function Survey({ navigation, route }) {

  const routeName = (route.params != undefined) ? route.params.name : '/survey' 

  const [isMainLoading, setIsMainLoading] = useState(true)
  const [heatmapModels, setHeapmapModels] = useState([])
  const [heatmapTypes, setHeatmapTypes] = useState([])
  
  useEffect(() => {
    fetchHeatmapData()
        .then(({ heatmapModelsNew, heatmapTypesNew }) => {
            console.log('heatmapModelsNew:',heatmapModelsNew)
            console.log('heatmapTypesNew:',heatmapTypesNew)
            setHeapmapModels(heatmapModelsNew);
            setHeatmapTypes(heatmapTypesNew);
            setIsMainLoading(false)
        })
        .catch(error => {
            console.error(error);
        });
  }, []);



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
        <Text style={styles.title}></Text>
      </View>)}
      <StatusBar style="auto" />
    </ScrollView>
  );
}
