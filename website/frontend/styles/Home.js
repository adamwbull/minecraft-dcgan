// Home.js styles

import { StyleSheet, Dimensions } from 'react-native';
import GlobalStyles from './GlobalStyles';
import { darkColors } from './Colors'

const screenWidth = Dimensions.get('window').width;

const LocalStyles = {
    container: {
        backgroundColor:darkColors.background,
        flex:1,
        width:'100%',
        alignItems:'center'
    },
    button: {
        padding:20,
        borderRadius:10,
        backgroundColor:darkColors.darkBlue,
        marginRight:screenWidth < 768 ? 0 : 20,
        marginTop:20,
        justifyContent:'center',
        alignItems:'center',
        width:screenWidth < 768 ? screenWidth-100 : 400
    },
    buttonText: {
        fontSize:20,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center'
    },
    title: {
        fontSize:24,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    description: {
        fontSize:20,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        marginTop:20,
    },
    landing: {
        padding:20,
        backgroundColor:darkColors.secondBackground,
        borderRadius:10,
        width:screenWidth < 768 ? screenWidth-60 : screenWidth/1.5,
        marginTop:20,
        marginBottom:20
    },
    row: {
        flexDirection:screenWidth < 768 ? 'column' : 'row',
        justifyContent:'center',
        alignItems:'center'
    }
};

// Merge global and local styles
const styles = { ...GlobalStyles, ...LocalStyles };

export default StyleSheet.create(styles);
