import { StyleSheet, Dimensions } from 'react-native';
import GlobalStyles from './GlobalStyles';
import { darkColors } from './Colors'

const screenWidth = Dimensions.get('window').width;

const LocalStyles = {
    container: {
        backgroundColor:darkColors.thirdBackground,
        flex:screenWidth < 768 ? null : 1,
        width:'100%'
    },
    datasetSelection: {
        flexDirection:screenWidth < 768 ? 'column' : 'row',
        justifyContent:'space-between',
        zIndex:10
    },
    topBar: {
        flexDirection:screenWidth < 768 ? 'column' : 'row',
        justifyContent:'flex-start',
        zIndex:10
    },
    arrowBox: {
        height:60,
        width:60,
        marginRight:20
    },
    homeText: {
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center'
    },
    topSection: {
        padding:20,
        backgroundColor:darkColors.secondBackground,
    },
    topSectionTitle: {
        fontWeight:'bold',
        color:darkColors.text,
        fontSize:22,
        fontFamily:'RobotoMono'
    },
    topSectionDesc: {
        marginBottom:10,
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    topSectionTools:screenWidth < 768 ? {
        flexDirection:'row-reverse',
        justifyContent:'flex-end',
        marginTop:0,
    } : {
        flexDirection:'row'
    },
    topSectionToolsAlternate:screenWidth < 768 ? {
        flexDirection:'row',
        marginTop:20,
        marginBottom:40
    } : {
        flexDirection:'row',
        marginLeft:10
    },
    topSectionTool: {
        alignItems:'flex-start',
        marginLeft:screenWidth < 768 ? 0 : 20
    },
    datasetDropdown: {
        marginLeft:screenWidth < 768 ? 0 : 20,
        width:250,
        marginRight:20,
        marginTop:screenWidth < 768 ? 20 : 0
    },
    tabRow: {
        justifyContent:'flex-start',
        alignItems:'center',
        flex:screenWidth < 768 ? null : 1,
    },
    tabButton: {
        padding:5,
        borderRadius:0,
        justifyContent:'center',
        alignItems:'center',
        borderWidth:5,
        marginRight:10
    },
    tabButtonSelected: {
        padding:5,
        borderRadius:0,
        borderWidth:5,
        justifyContent:'center',
        alignItems:'center',
        marginRight:10
    },
    tabButtonText: {
        textAlign:'center',
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    generateButton: {
        height:screenWidth < 768 ? 60 : 660,
        width:screenWidth < 768 ? screenWidth-40 : 60,
        justifyContent:'center',
        alignItems:'center'
    },
    generateButtonAlternate: { 
        flexDirection:'row', 
        justifyContent:'center',
        alignItems:'center', 
        transform: screenWidth < 768 ? null : [{ rotate: '270deg'}],
    },
    generateButtonText: {
        transform: screenWidth < 768 ? null : [{ rotate: '270deg'}],
        color:darkColors.text,
        fontSize:16,
        fontFamily:'RobotoMono',
        textAlign:'center',
        width:600
    },
    generateButtonTextSide: {
        color:darkColors.text,
        fontSize:16,
        fontFamily:'RobotoMono',
        marginRight:5
    },
    generationInfo: {
        flexDirection:'row',
        alignItems:'center',
        justifyContent:'space-between'
    },
    generationInfoText: {
        fontSize:screenWidth < 768 ? 14 : 16,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    leftSideButtons: {
        padding:20,
        backgroundColor:darkColors.secondBackground
    },
    toolText: {
        marginBottom:10,
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    rightSideButtons: {
        padding: 20,
        backgroundColor: darkColors.secondBackground,
        height:'100%'
    },
    rightSideButtonsInner: {
        justifyContent: 'space-between',
        alignItems: 'center',
        flexDirection:screenWidth < 768 ? 'row' : 'column',
        height:screenWidth < 768 ? null : 660,
    },
    carouselButton: {
        padding:10,
        backgroundColor:darkColors.darkBlue,
        justifyContent:'center',
        alignItems:'center',
        width:screenWidth < 768 ? null : '100%'
    },
    carouselButtonDisabled: {
        padding:10,
        backgroundColor:darkColors.midGray,
        justifyContent:'center',
        alignItems:'center',
        width:screenWidth < 768 ? null : '100%'
    },
    carouselButtonText: {
        color:darkColors.text,
        fontSize:16,
        fontFamily:'RobotoMono',
        textAlign:'center',
    },
    carouselButtonTextMiddle: {
        color:darkColors.text,
        fontSize:16,
        fontFamily:'RobotoMonoItalic',
        textAlign:'center',
    },
    input: {
        backgroundColor:'#fff',
        padding: 15,
        borderRadius: 10,
        color:darkColors.background,
    },    
    middleSection: {
        flexDirection:screenWidth < 768 ? 'column' : 'row',
        //flex:screenWidth < 768 ? null : 1,
    },  
    generationSection: { 
        backgroundColor:darkColors.background,
        flex:screenWidth < 768 ? null : 1,
        padding:20,
        //alignSelf: 'center', 
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent:'flex-start',
        alignItems: screenWidth < 768 ? 'flex-start' : 'flex-start',
    },
    generationContainerInner: {
        backgroundColor: darkColors.thirdBackground,
        borderRadius: 5,
        width: screenWidth < 768 ? screenWidth - 80 : 600,
        height: screenWidth < 768 ? screenWidth - 80 : 600,
        marginTop:10
    },     
    generationContainer: {
        paddingLeft:20,
        paddingRight:20,
        paddingTop:10,
        paddingBottom:20,
        backgroundColor:darkColors.secondBackground,
        alignSelf:'auto',
        borderRadius: 0,
        alignSelf: 'center'
    },
    loadMoreButtonWrapper: {
        justifyContent:'center',
        alignItems:'center'
    },
    loadMoreButton: {
        padding:20,
        borderRadius:0,
        backgroundColor:darkColors.darkBlue,
        justifyContent:'center',
        alignItems:'center'
    },
    loadMoreButtonText: {
        textAlign:'center',
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    blockStats: {
        display: 'flex', // Use flex layout
        flexDirection: 'row', // Lay out items in a row
        flexWrap: 'wrap', // Wrap items into new lines as needed
        width:screenWidth < 768 ? (screenWidth - 80) : 600
    },
    blockCountRow: {
        display: 'flex', // Use flex layout
        flexDirection: 'row', // Lay out items in a row
        flexWrap: 'wrap', // Wrap items into new lines as needed
        alignItems:'center',
        justifyContent:'space-between',
        width:screenWidth < 768 ? (screenWidth - 80)/2 : 300,
        paddingTop:20,
    },
    blockCountLeft: {
        flexDirection:'row',
        alignItems:'center',
    },
    blockCountDirectionBox: {
        paddingLeft:10,
    },
    blockCountDirectionBoxText: {
        fontSize:screenWidth < 768 ? 8 : 12,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    blockCountRight: {},
    blockCountText: {
        fontSize:screenWidth < 768 ? 10 : 16,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    statsTitle: {
        fontSize:screenWidth < 768 ? 16 : 20,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
};

// Merge global and local styles
const styles = { ...GlobalStyles, ...LocalStyles };

export default StyleSheet.create(styles);
