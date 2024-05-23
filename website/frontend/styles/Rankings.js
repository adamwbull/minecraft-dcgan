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
        width:screenWidth < 768 ? '100%' : '80%',
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
        width:screenWidth < 768 ? 25 : 70,
        height:screenWidth < 768 ? 25 : 70,
        justifyContent:'center',
        alignItems:'center'
    },
    heatmapSquareText: {
        fontSize:screenWidth < 768 ? 7 : 16,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center'
    },
    modelNameText: {
        fontSize:screenWidth < 768 ? 10 : 20,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center',
        marginLeft:10
    },
    heatmapTitleText: {
        fontSize:screenWidth < 768 ? 10 : 20,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center',
        transform: [{ rotate: '270deg'}]
    },
    title: {
        fontSize:screenWidth < 768 ? 14 : 24,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        marginBottom:20
    },
    heatmapTypes: {
        flexDirection:'row',
    },
    heatmapTitle: {
        width:screenWidth < 768 ? 25 : 70,
        height:screenWidth < 768 ? 40 : 100,
        marginTop:10
    },
    equation: {
        marginBottom:20
    },
    modelXplanation: {
        fontSize:14,
        color:darkColors.text,
        fontFamily:'RobotoMono',
    },
    equationImage: {
        marginBottom:10,
        marginTop:10
    },
    equationImageF: {
        width:screenWidth < 768 ? screenWidth-80 : null
    }
};

// Merge global and local styles
const styles = { ...GlobalStyles, ...LocalStyles };

export default StyleSheet.create(styles);
