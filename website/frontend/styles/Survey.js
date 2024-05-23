import { StyleSheet, Dimensions } from 'react-native';
import GlobalStyles from './GlobalStyles';
import { darkColors } from './Colors'

const screenWidth = Dimensions.get('window').width;

const LocalStyles = {
    container: {
        backgroundColor:darkColors.thirdBackground,
        flex:screenWidth < 768 ? null : 1,
        width:'100%',
        alignItems:'center'
    },
    middleSection: {
        padding:20,
        width:'80%',
        backgroundColor:darkColors.secondBackground,
    },
    row: {
        flexDirection:'row',
        alignItems:'center'
    },
    heatmapRow: {
        flexDirection:'row',
        alignItems:'center'
    },
    heatmapSquares: {
        flexDirection:'row',
        alignItems:'center'
    },
    heatmapSquare: {
        width:70,
        height:70,
        justifyContent:'center',
        alignItems:'center'
    },
    heatmapSquareText: {
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center'
    },
    modelNameText: {
        fontSize:20,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center',
        marginLeft:10
    },
    heatmapTitleText: {
        fontSize:20,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center',
        transform: [{ rotate: '270deg'}]
    },
    title: {
        fontSize:24,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        marginBottom:20
    },
    heatmapTypes: {
        flexDirection:'row',
    },
    heatmapTitle: {
        width:70,
        height:100,
        marginTop:10
    }
};

// Merge global and local styles
const styles = { ...GlobalStyles, ...LocalStyles };

export default StyleSheet.create(styles);
