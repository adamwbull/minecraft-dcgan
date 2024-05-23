import React, { useState, useEffect, useContext } from 'react';
import { Image, StatusBar, Text, View, Pressable, ActivityIndicator, ScrollView, Dimensions, Animated, TextInput, Switch  } from 'react-native';
import SchematicViewer from "../react-schematicwebviewer";
import { Buffer } from 'buffer';
import styles from '../styles/Models';
import { darkColors } from '../styles/Colors'
import { fetchModelGenerations, fetchNewModelGeneration, updateFavoriteStatus } from '../API'; 
import * as SplashScreen from 'expo-splash-screen';
import { AntDesign } from '@expo/vector-icons'
import DropDownPicker from 'react-native-dropdown-picker';
import Header from './shared/Header'
import userContext from '../Context.js'
import { useLinkTo } from '@react-navigation/native';

SplashScreen.preventAutoHideAsync();

//const jarUrl = "https://api-minecraftgan.adambullard.com/assets/faithful.zip"
const jarUrl = "http://localhost:8081/assets/client.jar"

const screenWidth = Dimensions.get('window').width;

window.Buffer = Buffer;

const blockImages = {
  'oak_stairs': require('../assets/blocks/oak_stairs.png'),
  'oak_slab': require('../assets/blocks/oak_slab.png'),
  'oak_log': require('../assets/blocks/oak_log.png'),
  'oak_planks': require('../assets/blocks/oak_planks.png'),
  'cobblestone': require('../assets/blocks/cobblestone.png'),
  'dirt': require('../assets/blocks/dirt.png'),
  'glass': require('../assets/blocks/glass.png'),
  'stone': require('../assets/blocks/stone.png'),
  'white_wool': require('../assets/blocks/white_wool.png'),
  'stone_bricks': require('../assets/blocks/stone_bricks.png'),
  'stone_brick_stairs': require('../assets/blocks/stone_brick_stairs.png'),
  'air': require('../assets/blocks/air.png'),
};

export default function Models({ navigation, route }) {

  const user = useContext(userContext)

  const linkTo = useLinkTo()

  const routeName = (route.params != undefined) ? route.params.name : '/models' 

  // Initialize userSettings as an object to store multiple settings
  const [userSettings, setUserSettings] = useState({});

  // Add state hooks
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [rankings, setRankings] = useState([]);
  const [newGeneration, setNewGeneration] = useState(null);
  const buttonColor = useState(new Animated.Value(0))[0];
  const [slideAnim] = useState(new Animated.Value(-600)); // Assuming the container slides in from the left

  const [displayCount, setDisplayCount] = useState(8); // Default count of generations to display
  const [carouselIndex, setCarouselIndex] = useState(0); // Index for the carousel

  const [currentModelIndex, setCurrentModelIndex] = useState([0, 0]);
  const [isMainLoading, setIsMainLoading] = useState(true); 
  const [moreSchematics, setMoreSchematics] = useState(true);
  const [totalSchemsForCurModel, setTotalSchemsForCurModel] = useState(0)

  const [favoritesOnly, setFavoritesOnly] = useState(false)
  const [showBlockStats, setShowBlockStats] = useState(false)

  // Load user settings on component mount
  useEffect(() => {

    //localStorage.clear()
    var savedSettings = undefined;

    if (typeof window !== 'undefined' && window.localStorage) {
      savedSettings = localStorage.getItem('userSettings');
      if (savedSettings) {
        savedSettings = JSON.parse(savedSettings);
      } else {
        savedSettings = {
          model_preference:[0, 0],
          display_count:3,
          favorites_only:false,
          show_block_stats:false
        }
        localStorage.setItem('userSettings', JSON.stringify(savedSettings));
      }
      setUserSettings(savedSettings);

    }

    ////console.log('savedSettings:',savedSettings)
    setDisplayCount(savedSettings.display_count)
    setFavoritesOnly(savedSettings.favorites_only)
    setShowBlockStats(savedSettings.show_block_stats)

    var modelIndexPreference = (savedSettings == undefined) ? undefined : savedSettings.model_preference
    loadMoreGenerations(undefined, 1, savedSettings.display_count, modelIndexPreference, savedSettings.favorites_only).finally(() => setIsMainLoading(false));

  }, []);

  // Function to save a specific setting by name
  const saveSetting = (settingName, value) => {
    const updatedSettings = { ...userSettings, [settingName]: value };
    setUserSettings(updatedSettings);
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem('userSettings', JSON.stringify(updatedSettings));
    }
  };

  const [averages, setAverages] = useState([])

  const loadMoreGenerations = async (targetModelName, page = 1, limit = 3, modelIndexPreference = undefined, favsOnly = false) => {

    setIsMainLoading(true);
    var [newGenerations, newModels, more, total, rankings, averageModelCounts] = await fetchModelGenerations(targetModelName, page, limit, favsOnly);

    console.log('modelIndexPreference:',modelIndexPreference)

    if (modelIndexPreference != undefined) {
      if (newModels[modelIndexPreference[0]].length < modelIndexPreference[1]) {
        modelIndexPreference[1] = 0
      }
      [newGenerations, newModels, more, total, rankings, averageModelCounts] = await fetchModelGenerations(newModels[modelIndexPreference[0]][modelIndexPreference[1]], page, limit, favsOnly);
      setCurrentModelIndex(modelIndexPreference)
      setRankingDropDownValue(modelIndexPreference[0])
      setModelDropDownValue(modelIndexPreference[1])
      //console.log('new data!',modelIndexPreference)
    }

    setRankings(rankings)
    setAverages(averageModelCounts)
    setMoreSchematics(more)
    setTotalSchemsForCurModel(total)

    if (page > 1) {
      setGenerations([...generations, ...newGenerations]);
    } else {
      setGenerations(newGenerations);
    }
    setModels(newModels);
    setIsMainLoading(false);
   
  };

  const simplifyTime = (time) => {
    // Parse the input string into a Date object
    const date = new Date(time);
    
    // Use toLocaleDateString to format the date in the desired format
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    const newTime = date.toLocaleDateString('en-US', options);
  
    return newTime;
  }

   // Generate sample and handle API call
  const generateSample = async () => {

    setIsLoading(true);
    setIsSuccess(false);
    ////console.log('generating with:',models[currentModelIndex[0]][currentModelIndex[1]])
    const result = await fetchNewModelGeneration(models[currentModelIndex[0]][currentModelIndex[1]], user.token)
    ////console.log('newGeneration:',result)
    setNewGeneration(result); // Update state with new sample data

    // Slide animation for the new generation container
    Animated.timing(slideAnim, {
      toValue: 0,
      duration: 500,
      useNativeDriver: false // Make sure to enable native driver for better performance
    }).start(() => {
      // After animation is done, add the new generation to the generations array and other relevant vars
      setTotalSchemsForCurModel(totalSchemsForCurModel+1)
      // Make sure we enable the next button if needed.
      if (generations.length > displayCount && generations.length < totalSchemsForCurModel+1) {
        setMoreSchematics(true)
      }
      setCarouselIndex(0)
      setGenerations(prevGenerations => [result, ...prevGenerations]);
      setNewGeneration(null); // Clear the newGeneration as it's now part of the generations
      slideAnim.setValue(-600); // Reset the position for the next use
    });

    setIsSuccess(true);
    // Animate button color to green
    Animated.timing(buttonColor, {
      toValue: 100,
      duration: 500,
      useNativeDriver: false
    }).start();

    // Reset button state after 2 seconds
    setTimeout(() => {
      setIsSuccess(false);
      setIsLoading(false);
      Animated.timing(buttonColor, {
        toValue: 0,
        duration: 500,
        useNativeDriver: false
      }).start();
      saveSetting("favorites_only", false)
      location.reload()
    }, 1000);

  };

  // Interpolate color value for button background
  const backgroundColor = buttonColor.interpolate({
    inputRange: [0, 100],
    outputRange: [darkColors.darkGreen, darkColors.green]
  });

  // Force SchematicViewer to re-render
  const [forceUpdateKey, setForceUpdateKey] = useState(0);

  const handleNext = () => {
    // Increment the forceUpdateKey to trigger a re-render
    setForceUpdateKey(prevKey => prevKey + 1);
    
    if ((carouselIndex + displayCount) >= generations.length) {
      const currentPage = Math.ceil(generations.length / displayCount);
      const nextPageToFetch = currentPage + 1;
      
      loadMoreGenerations(models[currentModelIndex[0]][currentModelIndex[1]], nextPageToFetch, displayCount, undefined, favoritesOnly).then(() => {
        setCarouselIndex(prevIndex => prevIndex + displayCount);
      });
    } else {
      setCarouselIndex(prevIndex => prevIndex + displayCount);
    }
  };

  const handlePrevious = () => {
    // Increment the forceUpdateKey to trigger a re-render
    setForceUpdateKey(prevKey => prevKey + 1);
    
    if (carouselIndex - displayCount >= 0) {
      setCarouselIndex(carouselIndex - displayCount);
    } else {
      setCarouselIndex(0);
    }
  };

  const updateDisplayCount = (text) => {


    const newDisplayCount = Number(text)

    // Set the variable.
    setDisplayCount(newDisplayCount)
    saveSetting('display_count', newDisplayCount)

    // Do we need to fetch more?
    if (generations.length < newDisplayCount) {

      // Fetch again.
      loadMoreGenerations(models[currentModelIndex[0]][currentModelIndex[1]], 1, newDisplayCount, undefined, favoritesOnly)

    }

    // No matter what, reset where we are on the carousel.
    setCarouselIndex(0)

  }

  const toggleFavoriteSchem = (id, newFavorite) => {

    // Update remote.
    updateFavoriteStatus(id, newFavorite).then(() => {
        //console.log('Favorite status updated successfully');
    }).catch(error => {
        console.error('Error updating favorite status:', error);
    });

    // Update local.
    var newGenerations = JSON.parse(JSON.stringify(generations));
    const index = newGenerations.findIndex(generation => generation.id === id);

    if (index !== -1) {
        newGenerations[index].favorite = newFavorite;
        setGenerations(newGenerations)
    } else {
        console.error('Generation not found with id:', id);
    }

  }

  const toggleFavoritesOnly = () => {

    // Fetch remote.
    loadMoreGenerations(models[currentModelIndex[0]][currentModelIndex[1]], 1, displayCount, undefined, !favoritesOnly)
    setCarouselIndex(0)
    // Update var.

    setFavoritesOnly(!favoritesOnly)
    saveSetting("favorites_only", !favoritesOnly)
    
  }

  const toggleBlockStats = () => {

    setShowBlockStats(!showBlockStats)
    saveSetting("show_block_stats", !showBlockStats)
  }

  const [generations, setGenerations] = useState([]);
  const [models, setModels] = useState([]);

  const [rankingDropdownOpen, setRankingDropDownOpen] = useState(false)
  const [rankingDropdownValue, setRankingDropDownValue] = useState(null)

  const [modelDropdownOpen, setModelDropDownOpen] = useState(false)
  const [modelDropdownValue, setModelDropDownValue] = useState(null)

  const [rankingItems, setRankingItems] = useState([]);
  const [modelItems, setModelItems] = useState([]);

  // Take in a state array [{custom data},...,n] and convert it to be an array of form [{label:'',value:''},..,n] for n entries
  const convertToDropdownItems = (stateArray, targetLabel) => {
    //console.log('stateArray:',stateArray)
    var result = stateArray.map((item, index) => ({
      label: item[targetLabel],
      value: index,
    }))
    //console.log('convertToDropdownItems:',result)
    return result;
  };
  
  const convertToDropdownItemsModels = (stateArray) => {
    var result = stateArray.map((item, index) => ({
      label: item,
      value: index,
    }))
    //console.log('convertToDropdownItems:',result)
    return result;
  };
  
  useEffect(() => {

    // Initialize rankings.
    const rankingsValues = convertToDropdownItems(rankings, 'name')
    setRankingItems(rankingsValues);
    console.log("rankingsValues:",rankingsValues)

  }, [generations]);

  const [prevRankingValue, setPrevRankingValue] = useState(null)
  
  useEffect(() => {

    if (rankingDropdownValue != null && models.length > rankingDropdownValue) {

      const selectedModels = models[rankingDropdownValue];
      console.log('selectedModels:',selectedModels)
      console.log('rankingDropdownValue:',rankingDropdownValue)
      const modelValues = convertToDropdownItemsModels(selectedModels);
      setModelItems(modelValues);
      if (prevRankingValue != null && prevRankingValue != rankingDropdownValue) {
        setModelDropDownValue(null)
      }
      setPrevRankingValue(rankingDropdownValue)
      //console.log("modelValues:",modelValues)

    }

  }, [models, rankingDropdownValue]);

  useEffect(() => {

    //console.log('rankingDropdownValue:',rankingDropdownValue)
    //console.log('modelDropdownValue:',modelDropdownValue)

    if (modelDropdownValue != null && rankingDropdownValue != null && modelDropdownValue != currentModelIndex[1]) {
      setCurrentModelIndex([rankingDropdownValue, modelDropdownValue]);
      saveSetting('model_preference', [rankingDropdownValue, modelDropdownValue])
      //await loadMoreGenerations(targetModelName, 1, displayCount, undefined, favoritesOnly);
      // We are going to location reload for now since SchematicViewer so buggy.
      location.reload()
    } 

  }, [modelDropdownValue])

  
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Header routeName={routeName} navigation={navigation} />
      <View style={[styles.topSection,{zIndex:10}]}>
        <View style={styles.topBar}>
          <View style={styles.modelSelection}>
            {false && (<View style={{width:300,zIndex:1000}}>
              <Text style={styles.topSectionDesc}>Ranking</Text>
              <DropDownPicker
                setOpen={setRankingDropDownOpen}
                open={rankingDropdownOpen}
                setValue={setRankingDropDownValue}
                value={rankingDropdownValue}
                setItems={setRankingItems}
                items={rankingItems}
                textStyle={styles.dropdownTextStyle}
                labelStyle={styles.dropdownLabelStyle}
              />
            </View>)}
            {rankingDropdownValue != null && (<View style={styles.modelDropdown}>
              <Text style={styles.topSectionDesc}>Model</Text>
              <DropDownPicker
                setOpen={setModelDropDownOpen}
                open={modelDropdownOpen}
                setValue={setModelDropDownValue}
                value={modelDropdownValue}
                setItems={setModelItems}
                items={modelItems}
                textStyle={styles.dropdownTextStyle}
                labelStyle={styles.dropdownLabelStyle}
              />
            </View>)}
          </View>
          <View style={styles.topSectionTools}>
            <View style={[styles.topSectionTool,{}]}>
              <Text style={styles.toolText}>Page Size</Text>
              <TextInput
                style={[styles.input,{width:100}]}
                onChangeText={text => updateDisplayCount(text)}
                value={String(displayCount)}
                inputMode="numeric"
              />
            </View>
            <View style={screenWidth < 768 ? [styles.topSectionTool,{marginRight:30}] : [styles.topSectionTool,{marginLeft:20}]}>
              <Text style={styles.toolText}>Faves Only</Text>
              <Pressable
                onPress={() => toggleFavoritesOnly()}
              >
                <Switch
                  value={favoritesOnly}
                  thumbColor={darkColors.midGray}
                  trackColor={darkColors.lightGray}
                  activeThumbColor={darkColors.darkGreen}
                  activeTrackColor={darkColors.green}
                  style={{
                    transform: [{ scaleX: 2 }, { scaleY: 2 }],
                    position:'absolute',
                    left:25,
                    top:15
                  }}
                />
              </Pressable>
            </View>
          </View>
          <View style={styles.topSectionToolsAlternate}>
            <View style={[styles.topSectionTool,{width:150}]}>
              <Text style={styles.toolText}>Block Stats</Text>
              <Pressable
                onPress={() => toggleBlockStats()}
              >
                <Switch
                  value={showBlockStats}
                  thumbColor={darkColors.midGray}
                  trackColor={darkColors.lightGray}
                  activeThumbColor={darkColors.darkGreen}
                  activeTrackColor={darkColors.green}
                  style={{
                    transform: [{ scaleX: 2 }, { scaleY: 2 }],
                    position:'absolute',
                    left:30,
                    top:15
                  }}
                />
              </Pressable>
            </View>
          </View>
        </View>
      </View>
      {isMainLoading ? (<ScrollView contentContainerStyle={[styles.middleSection,
        {
          justifyContent:'center',
          alignItems:'center'
        }
      ]}>

        <ActivityIndicator />

      </ScrollView>) : (<View style={styles.middleSection}>
        
        {user != null && (<View style={styles.leftSideButtons}>
          <Animated.View style={[styles.generateButton, { backgroundColor }]}>
            <Pressable onPress={() => { generateSample() }} type="button">
              {isSuccess ? (
                <View style={styles.generateButtonAlternate}>
                  <Text style={styles.generateButtonTextSide}>Success!</Text>
                  <AntDesign name="check" size={24} color="white" />
                </View>
              ) : isLoading ? (
                <View style={styles.generateButtonAlternate}>
                  <Text style={styles.generateButtonTextSide}>Generating...</Text>
                  <ActivityIndicator size="small" color="#fff" />
                </View>
              ) : (
                <Text style={styles.generateButtonText}>Generate Sample</Text>
              )}
            </Pressable>
          </Animated.View>
        </View>)}

        <View style={styles.generationSection}>

          {newGeneration && (
            <Animated.View
              style={[
                styles.generationContainer,
                { 
                  transform: [{ translateX: slideAnim }],
                  marginRight:20,
                  marginBottom:20
                }
              ]}
            >
              <View style={styles.generationInfo}>
                  <Text style={styles.generationInfoText}>{newGeneration.name}</Text>
                  <Text style={styles.generationInfoText}>{simplifyTime(newGeneration.created_at)}</Text>
                </View>
                <View style={styles.generationContainerInner}>
                  <SchematicViewer
                    orbit={false}
                    schematic={newGeneration.base64}
                    jarUrl={jarUrl}
                    loader={<View style={{
                      justifyContent:'center',
                      alignItems:'center',
                      flex:1,
                      width:600,
                      height:600
                    }}>
                      <ActivityIndicator />
                    </View>}
                    width={screenWidth < 768 ? screenWidth-120 : 600}
                    height={screenWidth < 768 ? screenWidth-120 : 600}
                    antialias={true}
                  />
                </View>
            </Animated.View>
          )}

          {generations.length > 0 && generations.slice(carouselIndex, carouselIndex + displayCount).map((generation, index) => {

            var generationContainerMarginValues = (index+1 == displayCount) ? {marginBottom:20} : {marginBottom:20,marginRight:20}
            const reversedIndex = totalSchemsForCurModel - (carouselIndex + index);
            
            console.log('generation:',generation)

            return (<View key={index} style={[styles.generationContainer,generationContainerMarginValues]}>
              <View style={styles.generationInfo}>
                <Text style={styles.generationInfoText}>#{reversedIndex}: {generation.name}</Text>
                <View style={{flexDirection:'row',alignItems:'center'}}>
                  <Text style={[styles.generationInfoText,{marginRight:10}]}>{simplifyTime(generation.created_at)}</Text>
                  <AntDesign 
                    name={generation.favorite ? 'star' : 'staro'} 
                    size={20} 
                    color={generation.favorite ? darkColors.darkGreen : darkColors.text} 
                    onPress={() => toggleFavoriteSchem(generation.id, generation.favorite ? 0 : 1)}
                  />
                </View>
              </View>
              <View style={styles.generationContainerInner} key={`${forceUpdateKey}-${index}`}>
                <SchematicViewer
                  orbit={false}
                  schematic={generation.base64}
                  jarUrl={jarUrl}
                  loader={<View style={{
                    justifyContent:'center',
                    alignItems:'center',
                    flex:1,
                    width:screenWidth < 768 ? screenWidth-80 : 600,
                    height:screenWidth < 768 ? screenWidth-80 : 600,
                  }}>
                    <ActivityIndicator />
                  </View>}
                  width={screenWidth < 768 ? screenWidth-80 : 600}
                  height={screenWidth < 768 ? screenWidth-80 : 600}
                  antialias={true}
                />
              </View>
              {showBlockStats && (<View style={styles.blockStatsWrapper}>
                <View style={{flexDirection:'row',alignItems:'center',justifyContent:'space-between'}}>
                  <Text style={styles.statsTitle}>Block Occurences</Text>
                  {screenWidth >= 768 && (<Text style={styles.blockCountText}>Structure / Dataset Avg</Text>)}
                </View>
                {generation.blockCounts.length > 0 && (<View style={styles.blockStats}>
                  {generation.blockCounts.map((block_count, block_count_index) => {

                    //console.log('block_count:',block_count)

                    const parseBlockId = (text) => {

                      // Remove the "minecraft:" prefix and split the block ID from its properties if any
                      let [blockName, properties] = text.replace('minecraft:', '').split('[');
                      blockName = blockName.trim(); // Ensure blockName is trimmed
                    
                      // Initialize direction and directionValue as null
                      let direction = null;
                      let directionValue = null;
                    
                      // If properties exist, parse the first relevant property
                      if (properties) {

                        properties = properties.slice(0, -1); // Remove the trailing ']'
                        const propertiesArray = properties.split(',');
                    
                        // Look for the first relevant property (axis, type, or facing)
                        for (let prop of propertiesArray) {
                          const [key, value] = prop.split('=');
                          if (key === 'axis' || key === 'type' || key === 'facing') {
                            direction = key;
                            directionValue = value;
                            //console.log(direction, directionValue)
                            break; // Stop at the first relevant property
                          }
                        }
                      }
                    
                      return [blockName, direction, directionValue];

                    };

                    [block_id, direction, directionValue] = parseBlockId(block_count.block_id)

                    const imageSource = blockImages[block_id]

                    var calculatedBlockRowPadding = (block_count_index % 2 == 0) ? {} : {paddingLeft:20}

                    // Find the average for the current block_id from the averages array
                    const averageObj = averages.find(avg => avg.block_id === block_count.block_id);
                    var average = averageObj ? averageObj.average_count : null;

                    // Function to color based on similarity and clean the average
                    const compareAverage = (average, count) => {
                      const ratio = average ? count / average : 0;
                      let color;

                      if (ratio <= 0.1) color = darkColors.red;
                      else if (ratio <= 0.5) color = darkColors.yellow;
                      else if (ratio <= 0.9) color = darkColors.lightGreen;
                      else if (ratio <= 1.3) color = darkColors.green; 
                      else if (ratio <= 1.8) color = darkColors.lightGreen;
                      else if (ratio <= 2.2) color = darkColors.yellow;
                      else color = darkColors.red;

                      //console.log('ratio:',ratio,count,average)

                      const cleanedAverage = Math.round(average);
                      return [color, cleanedAverage];
                    }

                    // Only proceed with compareAverage if average is not null
                    const [averageColor, cleanedAverage] = average ? compareAverage(average, block_count.count) : ['grey', 'N/A'];

                    return (<View 
                      key={'block_count_index_'+block_count_index+'-'+generation.name}
                      style={[styles.blockCountRow,calculatedBlockRowPadding]}
                    >
                      <View style={styles.blockCountLeft}>
                        <Image 
                          source={imageSource}
                          style={{width:20,height:20}}
                        />
                        {direction != null && (<View style={styles.blockCountDirectionBox}>
                          <Text style={styles.blockCountDirectionBoxText}>{direction}</Text>
                          <Text style={styles.blockCountDirectionBoxText}>{directionValue}</Text>
                        </View>)}
                      </View>
                      <View style={styles.blockCountRight}>
                       <Text style={styles.blockCountText}><Text style={{color:averageColor}}>{block_count.count}</Text> / {cleanedAverage}</Text>
                      </View>
                    </View>)
                  })}
                </View>) || (<View style={styles.blockStats}>
                  <Text style={styles.generationInfoText}>No block stats found, check back later.</Text> 
                </View>)}
              </View>)}
            </View>)

          })}

        </View>

        {generations.length > 0 && (<View style={styles.rightSideButtons}>
          <View style={styles.rightSideButtonsInner}>
            <Pressable disabled={carouselIndex == 0} onPress={handlePrevious} style={carouselIndex != 0 ? styles.carouselButton : styles.carouselButtonDisabled}>
              <Text style={styles.carouselButtonText}>Prev</Text>
            </Pressable>
            <Text style={styles.carouselButtonTextMiddle}>
              {`${totalSchemsForCurModel - carouselIndex} - ${Math.max(totalSchemsForCurModel - carouselIndex - displayCount + 1, 1)}`}
            </Text>
            <Pressable disabled={!moreSchematics} onPress={handleNext} style={moreSchematics ? styles.carouselButton : styles.carouselButtonDisabled}> 
              <Text style={styles.carouselButtonText}>Next</Text>
            </Pressable>
          </View>
        </View>)}


      </View>)}
      <StatusBar style="auto" />
    </ScrollView>
  );
}
