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

export default function Rankings({ navigation, route }) {

  const routeName = (route.params != undefined) ? route.params.name : '/rankings' 

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
        <Text style={styles.title}>KL Divergence Heatmap</Text>
        <View style={styles.row}>
          <View style={{flex:1}}>
            <View style={styles.heatmapWrapper}>
              {heatmapModels.map((model, index) => {

                return (<View style={styles.heatmapRow} key={'modelrow-'+index}>
                  <View style={styles.heatmapSquares}>
                    {model.scores.map((square, scoreIndex) => {
                      return (<View style={styles.heatmapSquareWrapper} key={'score-'+scoreIndex}>
                        <View style={[styles.heatmapSquare,{backgroundColor:square.color}]}>
                          <Text style={styles.heatmapSquareText}>{Math.round(square.score * 100) / 100}</Text>
                        </View>
                      </View>)
                    })}
                  </View>
                  <View style={styles.modelName}>
                    <Text style={styles.modelNameText}>{model.name}</Text>
                  </View>
                </View>)
              })}
              <View style={styles.heatmapTypes}>
                {heatmapTypes.map((heatmapType, heatmapTypeIndex) => {
                  return (<View style={styles.heatmapTitle} key={'heatmapTypeIndex-'+heatmapTypeIndex}>
                    <Text style={styles.heatmapTitleText}>{heatmapType}</Text>
                  </View>)
                })}
              </View>
            </View>
          </View>
          <View style={{flex:1}}>
            <Text style={styles.title}>KL Divergence Heatmap</Text>
          </View>
        </View>
        <Text style={styles.title}>KL Divergence Explanation</Text>
        <View style={{marginBottom:20}}>
          <Text style={styles.modelXplanation}>w0.0: novel generated patterns are penalized, but no penalty for not using dataset patterns</Text>
          <Text style={styles.modelXplanation}>w1.0: penalty for not using dataset patterns, novel generated patterns are not penalized</Text>
          <Text style={styles.modelXplanation}>T^3: indicates KL measured by extracting patterns from samples of dim T</Text>
          <Text style={styles.modelXplanation}>F: output in a heatmap cell for a given w, T</Text>
        </View>
        <View>
          <Text style={styles.modelXplanation}>
            F(P, Q) evaluates the fitness of a set of generated structures Q against a set of sample levels P, combining weights w and (1-w) for each direction of KL divergence.
          </Text>
          <View style={styles.equation}>
            <Image style={[styles.equationImage,styles.equationImageF]} source={require('../assets/Fitness_F.png')} resizeMode="contain" />
          </View>
          <Text style={styles.modelXplanation}>
            The equation calculates the expected log difference between two probability distributions P and Q over 3D patterns.
          </Text>
          <View style={styles.equation}>
            <Image style={[styles.equationImage,styles.equationImageDKL]} source={require('../assets/D_KL.png')} resizeMode="contain" />
            <Text style={styles.modelXplanation}>X: set of all patterns found in both P and Q</Text>
          </View>
          <Text style={styles.modelXplanation}>
            However, P' and Q' are derived to adjust P and Q to prevent division by zero, using an epsilon correction for stability.
          </Text>
          <View style={styles.equation}>
            <Image style={styles.equationImage} source={require('../assets/modified_P_prime.png')} resizeMode="contain"  />
          </View>
          <Text style={styles.modelXplanation}>
            The divergence is then used to measure how similar generated structures are to sample levels across variations of w, T.
          </Text>
        </View>
      </View>)}
      <StatusBar style="auto" />
    </ScrollView>
  );
}
