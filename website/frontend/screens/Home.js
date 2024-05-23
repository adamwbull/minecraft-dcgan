import React, { useState } from 'react';
import { View, Text, Pressable, Linking, ScrollView } from 'react-native';
import styles from '../styles/Home';
import { updateBlockDistributions, updatePatternDistributions } from '../API'
import Header from './shared/Header'
import { useLinkTo } from '@react-navigation/native';

function Home({ navigation, route }) {

  const routeName = '/' 
  const linkTo = useLinkTo()

  const [blockDistributionsUpdating, setBlockDistributionsUpdating] = useState(false)

  const triggerBlockDistributions = async () => {

    setBlockDistributionsUpdating(true)

    const update = await updateBlockDistributions()

    setBlockDistributionsUpdating(false)
    
  }

  const [patternDistributionsUpdating, setPatternDistributionsUpdating] = useState(false)

  const triggerPatternDistributions = async () => {

    setPatternDistributionsUpdating(true)

    const update = await updatePatternDistributions(1, 2)

    setPatternDistributionsUpdating(false)
    
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Header routeName={routeName} navigation={navigation} />
      <View style={styles.landing}>
        <Text style={styles.title}>Minecraft Structure Generator</Text>
        <Text style={styles.description}>
          Leveraging Deep Convolutional Generative Adversarial Networks (DCGANs)
        </Text>
        <Text style={styles.description}>
          Read our paper, view the source code, or start creating structures right away:
        </Text>
        <View style={styles.row}>
          <Pressable
            onPress={() => Linking.openURL('https://github.com/deneke-research/minecraft-gan-website/paper.pdf')}
            style={({ pressed }) => [{ backgroundColor: pressed ? '#ddd' : '#00f' }, styles.button]}
          >
            <Text style={styles.buttonText}>Read the Paper</Text>
          </Pressable>
          
          <Pressable
            onPress={() => Linking.openURL('https://github.com/deneke-research/minecraft-gan-website')}
            style={({ pressed }) => [{ backgroundColor: pressed ? '#ddd' : '#f90' }, styles.button]}
          >
            <Text style={styles.buttonText}>View Source Code</Text>
          </Pressable>
          <Pressable
            onPress={() => linkTo('/models')}
            style={({ pressed }) => [{ backgroundColor: pressed ? '#ddd' : '#0f0' }, styles.button]}
          >
            <Text style={styles.buttonText}>Get Started</Text>
          </Pressable>
        </View>
      </View>
    </ScrollView>
  );

}

export default Home;