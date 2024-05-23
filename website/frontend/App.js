import React, { useCallback } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useFonts } from 'expo-font';

import { darkColors } from './styles/Colors'

// Pages.
import Home from './screens/Home';
import Models from './screens/Models';
import Dataset from './screens/Dataset';
import Rankings from './screens/Rankings';
import Upload from './screens/Upload';
import Login from './screens/Login';
import Profile from './screens/Profile';

const Stack = createNativeStackNavigator();

// Default navbar locations.
export const locations = [
  {
    name: "Home",
    routeName:"/home",
    loginRequired:false,
    minLoginTypeRequired:-1
  },
  {
    name: "Models",
    routeName:"/models",
    loginRequired:false,
    minLoginTypeRequired:-1
  },
  {
    name: "Dataset",
    routeName:"/dataset",
    loginRequired:true,
    minLoginTypeRequired:1
  },
  {
    name: "Upload",
    routeName:"/upload",
    loginRequired:true,
    minLoginTypeRequired:1
  },
  {
    name: "Rankings",
    routeName:"/rankings",
    loginRequired:false,
    minLoginTypeRequired:-1
  }
]

const linking = {
  prefixes: ['https://minecraftgan.adambullard.com/', 'http://localhost:8081/'],
  config: {
    screens: {
      Home: 'home',
      Models: {
        path: 'models/:name?', 
        parse: {
          id: (name) => `${name}`, // Convert id to string
        },
      },
      Dataset: 'dataset',
      Rankings: 'rankings',
      Upload: 'upload',
      Login: 'login',
      Profile: 'profile'
    },
  },
};

function App() {

  const [fontsLoaded, fontError] = useFonts({
    'RobotoMono': require('./assets/fonts/RobotoMono.ttf'),
    'RobotoMonoItalic': require('./assets/fonts/RobotoMonoItalic.ttf')
  });

  const onLayoutRootView = useCallback(async () => {
    if (fontsLoaded || fontError) {
      await SplashScreen.hideAsync();
    }
  }, [fontsLoaded, fontError]);

  return (
    <NavigationContainer linking={linking} onLayout={onLayoutRootView}> 
      <Stack.Navigator initialRouteName="Home"
        screenOptions={{
          headerShown: false,
          contentStyle: {
            backgroundColor:darkColors.background
          }
        }}
      >
        <Stack.Screen name="Home" component={Home} />
        <Stack.Screen name="Models" component={Models} />
        <Stack.Screen name="Dataset" component={Dataset} />
        <Stack.Screen name="Rankings" component={Rankings} />
        <Stack.Screen name="Upload" component={Upload} />
        <Stack.Screen name="Login" component={Login} />
        <Stack.Screen name="Profile" component={Profile} />
      </Stack.Navigator>
    </NavigationContainer>
  );
  
}

export default App;
