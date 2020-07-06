'''
MIMIC2's data cleaning & feature engineering

This data processing script is long, and the logic is very much convoluted. 
Put simply, the input data comes in in the format of typical log files: patient ID, timestamp, event, event value, and
we want to generate the standard, patient-feature matrix of dimension N by K. N = number of patients, K = number of features.

There are many steps we took:

1. Read in all logs chronologically, get a list of unique "events" observed in the dataset

2. Not all "events" become features, decide which will be included as features (remove rare events etc..)

3. For each MIMIC dataset, we will end up with thousands of features, but for our model, categorical and continuous features
   are handled differently. Therefore, we need to categorize features into different groups, and this is done by
   looking at observed values for a specific feature, and (often times) hardcoding some rules.

4. Each patient can have multiple event values under the same event, so we need to encode lists of values 
   (take max/min/median, or one-hot encode by count etc.)

5. Encode missingness, impute with zero, and add missing flags if necessary

During development of the steps above, I had to print out a lot of stuff, hard coded edge cases, and definitly
made the code difficult to read as a consequence. Rerunning will not give any bugs, but it is likely that 
modifying/extending the code could be a real pain. If later becomes necessary, my suggestion is that the user 
carefully read through the entire script before trying to fix anything, or adding any ad-hoc components. 

'''

import copy, argparse, os, datetime
import re, statistics, pickle, sys, collections
import pandas as pd 
import numpy as np

from preprocessing_mimic2_lib.FeatureClasses import ICUEventSet
from preprocessing_mimic2_lib.FeatureUtils import clean1, clean2, clean3, clean4,\
     clean5, clean6, parseBloodGasVentilationRate

from fancyimpute import SoftImpute

class FeatureEngineerNew2():

    def __init__(self, verbose=True, dataPath=None):
        '''
        Define basic class attributes.

        '''
        self.verbose = verbose
        # name of data sources
        self.SOURCES = ["pancreatitis", "ich", "sepsis"]
        self.dataPath = dataPath

        if dataPath is None:
            self.DATA_PATH = "preprocessed_data/" + "{SOURCE}_preprocessed_full/{FILENAME}"
        else:
            self.DATA_PATH = dataPath + "{SOURCE}_preprocessed_full/{FILENAME}"

    def _GetFilePath(self, sourceIndex, filename):
        """
        Helper function, get the real path
        :param source_index: 0 -> "pancreatitis", 1 -> "ich", 2 -> "sepsis"
        :param filename: "train.csv", "test.csv", ...
        :return: an absolute file path
        """
        return self.DATA_PATH.format(SOURCE=self.SOURCES[sourceIndex],
                                     FILENAME=filename)

    def Fit(self, sourceIndex, filePrefix='', \
            rareFrequency=0.05, rareFrequencyCategory=0.05):
        '''
        Identifies all features form training data, and group them in one of the 
        following categories: 1. Categorical; 2. Continuous; 3. Indicator; 4. All-Na

        :Requires a train.csv stored at the specified location

        '''
        fname = filePrefix + '{}.csv'
        # trainDataframe = pd.read_csv(self._GetFilePath(sourceIndex,
        #                              fname.format('train')), header=None, \
        #                              names=["patientID", "event", "value"])
        trainDataframe = pd.read_csv(self._GetFilePath(sourceIndex,
                                     fname.format('full')), header=None, \
                                     names=["patientID", "event", "value"])

        self.trainIds = list(trainDataframe['patientID'].unique())
        self.totalNumPatients = len(self.trainIds)

        self.rareFrequency = rareFrequency
        self.rareFrequencyCategory = rareFrequencyCategory
        self.sourceIndex = sourceIndex

        self.data = trainDataframe

        if self.verbose: print('[FEATURE] Finish loading EHR.')

        # # # TEST
        # self.__testVisualizeFeaturesList = []

        self._UnderstandEvents()
        # self._UnderstandEventsUnitTest()
        if self.verbose:
            print('[FEATURE] Finished categorizing all observed events.')
        # self._PrintFeatureList0()

        # # # TEST
        # np.savetxt('sepsis_features.csv', \
        #     np.array(self.__testVisualizeFeaturesList), delimiter = ",", \
        #     header = "eventName, eventValList, nVals, eventValSet, nUniqueVals, eventCategory, allNa", fmt = "%s")

    def FitTransform(self, filePrefix='', discretized=False, onehot='default', \
                     zerofilling=True):
        '''
        Transform either training or testing data according to the categorized events.

        :params: filePrefix: specifies a file location
        :params: discretized: whether the output dataframe contains only discrete
                              values.
        :params: onehot: 'default': number of one-hot encodings = number of classes
                'reference': removes one of the one-hot encodings as a reference class

        :returns: self.patientRecords: a pandas dataframe

        '''
        # loading training data 
        self._LoadEventsToPatients('train')
        self._PrintFeatureList1() 
        if self.verbose:
            print('\n')
            print('[FEATURE] Finished loading training events to patients.')

        # no need to load testing data, all data now goes to train
        self.testIds = []
        self.dataToFit = None

        # # loading testing data
        # fname = filePrefix + '{}.csv'
        # testDataframe = pd.read_csv(self._GetFilePath(self.sourceIndex,
        #                             fname.format('test')), header=None, \
        #                             names=["patientID", "event", "value"])

        # self.testIds = list(testDataframe['patientID'].unique())
        # self.dataToFit = testDataframe
        # self._LoadEventsToPatients('test')
        # if self.verbose:
        #     print('\n')
        #     print('[FEATURE] Finished loading testing events to patients.')

        # # TEST
        # Save current feature list for analysis
        # self.__testSaveFeaturesAfterFit()

        # # TEST
        # Save current data cleaning object in a pickle file to continue progress later
        # pickle.dump(self, open("sepsis_self.pickle", "wb"))

        # Note that we aggregate observations for training and testing patients 
        # simultaneously
        self._AggregateObservations(discretized, onehot, zerofilling)
        if self.verbose:
            print('[FEATURE] All observation series collapsed to one')        
        self._PrintFeatureList1()

        self.ImputeMissing(zerofilling)
        if self.verbose:
            print('[FEATURE] All missing entries imputed.')

        return

    def __testSaveFeaturesAfterFit(self):
        '''
        Save all features, their category, and observed values (in both training 
        and testing data), as both list and set.

        Used for testing purposes.

        '''
        allFeatureInfo = dict()

        for patientID in self.patientRecords:
            for patientEvent in self.patientRecords[patientID]:
                if patientEvent not in allFeatureInfo:   # init
                    currEventCategory = self.events.allEventCategories[patientEvent]
                    if currEventCategory == 'categorical' and \
                        patientEvent in self.events.categoricalTypes:
                        currEventInfo = self.events.CategoricalType(patientEvent)
                    else:
                        currEventInfo = None
                    allFeatureInfo[patientEvent] = \
                        [patientEvent, currEventCategory, currEventInfo, list(), None]

                allFeatureInfo[patientEvent][3].extend(self.patientRecords[patientID][patientEvent])

        resultFeatureInfo = []
        for feature in allFeatureInfo:
            currInfo = []
            for token_i in range(5):
                if token_i == 4:
                    token = set(allFeatureInfo[feature][token_i - 1])
                else:
                    token = allFeatureInfo[feature][token_i]
                currInfo.append(str(token).replace(",", "."))
            resultFeatureInfo.append(currInfo)

        fileName = '{}_saved_features.csv'.format(self.SOURCES[self.sourceIndex])
        np.savetxt(fileName, np.array(resultFeatureInfo), delimiter=',',\
            header="eventName, eventCategory, additionalInfo, valueList, valueSet", fmt="%s")
        print("[TESTING] Feature details saved to {}".format(fileName))


    def _UnderstandEvents(self):
        '''
        Read EHR row by row, get all observed events and their observed values.
        Categorize events into the following categories:
            - Categorical 
            - Continuous
            - Indicator
            - All NA

        Reads from:
            self.data: pandas.core.frame.DataFrame with 3 columns
                       ["patientID", "event", "value"]
                       This dataframe only contains training data.

        Modifies:
            self.events: ICUEventSet object

        '''
        self.events = ICUEventSet(self.totalNumPatients, self.rareFrequency,\
                                  self.rareFrequencyCategory)

        eventNames = self.data['event'].unique()
        for eventName in eventNames:

            eventRemove = False
            eventValuesWithDup = \
                        self.data[self.data['event'] == eventName]['value']
            eventNameNew = clean1(eventName)
            eventValuesWithDup, eventCount, allNa, NanWarning = \
                                                    clean2(eventValuesWithDup)

            # if NanWarning:
            #     print('[WARNING] Found NAN values among original event values, for event:{}'.format(eventName))
            #     if allNa:
            #         print('[WARNING] This is an all-na event.')
            #     print('\n')

            # if allNa:
            #     print('[Debug] Found all-na event: {}'.format(eventNameNew))

            if not allNa:
                # remove infrequent events that are not already one-hot encoded
                if eventCount/self.totalNumPatients <= self.rareFrequency:
                    eventRemove = True 

            if not eventRemove:             
                self.events.AddEvent(eventNameNew, eventValuesWithDup, allNa)

                # if eventName.endswith(":"):
                    # print("[WARNING] Found events ending with colons: {}. Special handling required.".format(eventName))
                    # self.events.RegisterEventEndingWithColons(eventNameNew)

                # self.__testVisualizeFeatures(eventNameNew, eventValuesWithDup, allNa)

        # self.events.AnalyzeEventsEndingWithColons()

        # After understanding all events, need to call the following function
        # to collectively analyze allNa events.
        self.events.AnalyzeAllNaEvents()

        # Make a copy of the orginal events (before loading records to patients)
        self.eventsUnfitted = self.events.MakeCopy()

    def __testVisualizeFeatures(self, eventName, eventValuesWithDup, allNa):
        self.__testVisualizeFeaturesList.append(\
            [eventName.replace(",", "."), \
             str(eventValuesWithDup).replace(",", "."), \
             len(eventValuesWithDup), \
             str(set(eventValuesWithDup)).replace(",", "."),\
             len(set(eventValuesWithDup)), \
             self.events.allEventCategories[eventName], allNa])

    def _PrintFeatureList0(self):
        '''
        Prints current composition of seen events.

        '''
        eventSummary = self.events.GetSummary()

        if self.verbose:

            print('\n')
            print('  [Summary] Feature summary up to this step:')
            print('    - Total features: {}'.format(eventSummary[0]))
            print('    - Indicators: {}'.format(eventSummary[1]))
            print('    - Categorical: {}'.format(eventSummary[2]))
            print('    - Continuous: {}'.format(eventSummary[3]))
            print('    - All NA: {}'.format(eventSummary[4]))
            print('\n')

    def _PrintFeatureList1(self):
        '''
        After loading all events to patients.

        '''
        # Update event bookkeeping to get the fitted events       
        for newEvent in self.newRecords:
            if newEvent not in self.events.allEventNames:
                self.events.allEventNames.add(newEvent)
            self.events.allEventCategories[newEvent] = self.newRecords[newEvent]

        for delEvent in self.deletedRecords:
            self.events.allEventNames.remove(delEvent)
            self.events.allEventCategories.pop(delEvent, None)

        self._PrintFeatureList0()

    def _LoadEventsToPatients(self, mode = 'train'):
        '''
        Build self.patients, a dictionary (key = patientID) of dictionary 
        (key = eventName).

        Modifies:
            self.patientRecords

        '''
        # constants & lookup table
        positiveSet = {'yes', 'done', '1', '1.0'}
        negativeSet = {'no', 'not_done', '0', '0.0'}
        naSet = {'na', 'na*', 'other'}

        # Configure loading mode
        self.mode = mode
        if self.mode == 'train':
            # for patient records
            self.patientRecords = dict()
            self.newRecords = dict()     # new event name: (category, possible vals)
            self.deletedRecords = set()
            currData = self.data
            
            # keeping track of some global info
            self.categoricalEventRoots = dict()   # for one-hot encoding
            self.continuousEventVals = dict()     # for binning
            self.indicatorEventVals = dict()     # for differentiating different indicators

        elif self.mode == 'test':
            currData = self.dataToFit

        else:
            raise NotImplementedError

        rawEventNames = self.data['event'].unique() # ONLY from training data
        progressCount = 0
        totalCount = len(rawEventNames)

        for rawEventName in rawEventNames:

            ###################################################################
            # Prints progress bar
            # progressCount += 1
            # if progressCount % 400 == 0:
            #     print('[PROGRESS] Processing {} out of {} events, status = {}'.format(\
            #                     progressCount, totalCount, progressCount/totalCount))
            ###################################################################

            rawEventRecords = currData[currData['event'] == rawEventName]
            eventName = clean1(rawEventName)

            if self.eventsUnfitted.Contains(eventName):         # excluding rare events
                for rawEventRecord in rawEventRecords.values:
                    eventCategory = self.eventsUnfitted.allEventCategories[eventName]
                    patientID = rawEventRecord[0]

                    if eventCategory == 'categorical':
                        rawEventValue = rawEventRecord[2]
                        eventValue = clean6(eventName, rawEventValue)    # edge
                        eventType = self.eventsUnfitted.CategoricalType(eventName)

                        if eventName == ['lab:blood:hematology:anisocytosis', 'lab:blood:hematology:poikilocytosis']:
                            print(eventName, eventValue, eventType)
                            print("))))))))))))))))))")


                        # First handle a handful of edge cases:                        
                        if eventName == 'lab:blood:blood_gas:ventilation_rate':
                            # special case
                            leftVal, rightVal = parseBloodGasVentilationRate(\
                                                              rawEventRecord[2])
                            if leftVal != None:
                                modifiedEventName = eventName + ':left_value'
                                self._AddRecord(patientID, \
                                               modifiedEventName, leftVal,\
                                               modifiedEvent=True, \
                                               modifiedCategory='continuous',\
                                               deleteOld=eventName)
                                self._LogContinuous(modifiedEventName, leftVal)

                            if rightVal != None:
                                modifiedEventName = eventName + ':right_value'
                                self._AddRecord(patientID, \
                                               modifiedEventName, rightVal, \
                                               modifiedEvent=True,\
                                               modifiedCategory='continuous',\
                                               deleteOld=eventName)
                                self._LogContinuous(modifiedEventName, rightVal)

                        # elif eventName == 'chart:diagnosis/op:na':
                        #     # special case
                        #     if 'abdominal' in eventValue:
                        #         category = 'abdominal_related'
                        #     elif 'etoh' or 'pancr' in eventValue:
                        #         category = 'pancreatitis_related'
                        #     else:
                        #         category = 'others'

                        #     self._AddRecord(patientID, eventName, category)
                        #     self._LogCategorical(patientID, base=eventName,\
                        #                                      category=category)

                        # Then handle using the instructions returned in 'eventType'
                        elif eventType == None:
                            category = eventValue
                            self._AddRecord(patientID, eventName, category)
                            self._LogCategorical(patientID, base=eventName,\
                                                             category=category)

                        elif type(eventType) == type(set()):
                            # merge rare categories
                            if eventValue in eventType: 
                                category = 'merged_others_' + str(eventType)
                            else:
                                category = eventValue
                            self._AddRecord(patientID, eventName, category)
                            self._LogCategorical(patientID, base=eventName,\
                                                             category=category)

                        elif type(eventType) == type(list()) and eventValue in eventType:
                            # leveled categorical 
                            eventValue = eventType.index(eventValue)
                            self._AddRecord(patientID, eventName, eventValue, \
                                           modifiedEvent=True, \
                                           modifiedCategory='continuous')
                            self._LogContinuous(eventName, eventValue)

                        elif type(eventType) == type(dict()) and eventValue in eventType:
                            # leveled categorical
                            # currently does not support null handling
                            eventValue = eventType[eventValue]
                            self._AddRecord(patientID, eventName, eventValue, \
                                           modifiedEvent=True, \
                                           modifiedCategory='continuous')
                            self._LogContinuous(eventName, eventValue)

                        elif type(eventType) == str:
                            # should have been classified to other category
                            # drop down to the other cases
                            eventCategory = eventType

                        else:
                            print('[WARNING] Found CATEGORICAL event' + \
                                      'with unrecognized value, dropped record.')
                            print(eventName, eventValue)
                            # raise NotImplementedError

                    if eventCategory == 'continuous':
                        rawEventValue = rawEventRecord[2]
                        eventValue = clean3(rawEventValue)          # clean neg
                        eventValue = clean4(eventValue)           # clean units
                        eventValue = clean5(eventName, eventValue) # edge cases

                        if eventValue == 'na':
                            # create an indicator for this event
                            # event + na: --> 1 
                            modifiedEventName = eventName + ':continuous_na'
                            self._AddRecord(patientID, modifiedEventName, 1,\
                                           modifiedEvent=True, \
                                           modifiedCategory='indicator',\
                                           handleInstructions='fill')
                            self._LogIndicator(modifiedEventName, 1)

                        elif eventValue == 'tr':
                            # create an indicator for this event
                            # event + tr: --> 1
                            modifiedEventName = eventName + ':continuous_tr'
                            self._AddRecord(patientID, modifiedEventName, 1,\
                                           modifiedEvent=True,\
                                           modifiedCategory='indicator')
                            self._LogIndicator(modifiedEventName, 1)

                        else:

                            try:
                                self._AddRecord(patientID, eventName, \
                                               float(eventValue), \
                                               modifiedEvent=True, \
                                               modifiedCategory='continuous')
                                self._LogContinuous(eventName, float(eventValue))

                            except:
                                print('[WARNING] Found CONTINUOUS event' + \
                                      'with unrecognized value, encoded as NaN')
                                print(eventName, eventValue)

                                modifiedEventName = eventName + ':continuous_na'
                                self._AddRecord(patientID, modifiedEventName, 1,\
                                               modifiedEvent=True,\
                                               modifiedCategory='indicator')
                                self._LogIndicator(modifiedEventName, 1)

                    if eventCategory == 'indicator':
                        rawEventValue = rawEventRecord[2]
                        eventValue_, _, _, _ = clean2([rawEventValue])
                        eventValue = eventValue_[0]

                        if eventValue in positiveSet:
                            modifiedEventName = eventName + ':+/-'
                            self._AddRecord(patientID, modifiedEventName, 1,\
                                            modifiedEvent=True, \
                                            modifiedCategory='indicator',\
                                            deleteOld=eventName)
                            self._LogIndicator(modifiedEventName, 1)

                        elif eventValue in negativeSet:
                            modifiedEventName = eventName + ':+/-'
                            self._AddRecord(patientID, modifiedEventName, 0,\
                                            modifiedEvent=True, \
                                            modifiedCategory='indicator',\
                                            deleteOld=eventName)
                            self._LogIndicator(modifiedEventName, 0)

                        elif eventValue in naSet:
                            modifiedEventName = eventName + ':indicator_na'
                            self._AddRecord(patientID, modifiedEventName, 1,\
                                           modifiedEvent=True,\
                                           modifiedCategory='indicator',\
                                           deleteOld=eventName)
                            self._LogIndicator(modifiedEventName, 1)

                        else:
                            print('[WARNING] unrecognized indicator values, encoded as NaN')
                            print(eventName, eventValue, type(eventValue))
                            modifiedEventName = eventName + ':indicator_na'
                            self._AddRecord(patientID, modifiedEventName, 1,\
                                           modifiedEvent=True,\
                                           modifiedCategory='indicator',\
                                           deleteOld=eventName)
                            self._LogIndicator(modifiedEventName, 1)

                    if eventCategory == 'all_na':
                        # reverse one-hot encoding
                        eventType = self.eventsUnfitted.AllNaType(eventName)

                        if type(eventType) == type(None):
                            base = ":".join(eventName.split(':')[:-1])
                            category = eventName.split(':')[-1]
                            self._AddRecord(patientID, base, category, \
                                            modifiedEvent=True, \
                                            modifiedCategory='categorical', \
                                            deleteOld=eventName)
                            self._LogCategorical(patientID, base, category)

                        elif type(eventType) == str:
                            if eventType == 'drop':
                                self.deletedRecords.add(eventName)

                            elif eventType in ['indicator', 'timestamps']:
                                self._AddRecord(patientID, eventName, 1,\
                                               modifiedEvent=True,\
                                               modifiedCategory='indicator')
                                self._LogIndicator(eventName, 1)
                            else:
                                print(\
                                '[Debug] Unrecognized all_na event {}'.format(\
                                                                    eventName))
                                raise NotImplementedError

                        elif type(eventType) == type(set()):
                            base = ":".join(eventName.split(':')[:-1])
                            if eventName in eventType:
                                category = 'merged_others_' + str(eventType)
                            else:
                                category = eventName.split(":")[-1]

                            self._AddRecord(patientID, base, category,\
                                           modifiedEvent=True,\
                                           modifiedCategory='categorical',
                                           deleteOld=eventName)
                            self._LogCategorical(patientID, base, category)

                        else:
                            print('[Debug] Unrecognized all_na event {}'.format(\
                                                                      eventName))
                            raise NotImplementedError

                    if eventCategory not in ['categorical', 'continuous',\
                                             'indicator', 'all_na']:
                        print('[Debug] Unrecognized event type.{}'.format(\
                                                                eventCategory))
                        raise NotImplementedError

    def _AddRecord(self, patientID, eventName, eventValue,\
                  modifiedEvent=False, modifiedCategory=None, \
                  handleInstructions=None, deleteOld=None):
# ---------------------------------------

        if self.mode == 'test':
            # remove unseen events
            if eventName not in self.events.allEventNames:
                return
            # handles unseen categories
            if type(eventValue) == str:
                base = eventName
                category = eventValue
                assert(base in self.categoricalEventRoots)
                allCategories = self.categoricalEventRoots[base]
                if category not in allCategories:
                    eventValue = None
                    for allCategory in allCategories:
                        if "merged_others" in allCategory:
                            eventValue = allCategory
                if eventValue == None:
                    return

        # 1. Write in patient records
        # ------------------------
        if patientID not in self.patientRecords:
            self.patientRecords[patientID] = dict()

        if eventName not in self.patientRecords[patientID]:
            self.patientRecords[patientID][eventName] = []

        self.patientRecords[patientID][eventName].append(eventValue)

        if self.mode == 'test': return # testing subjects do not add new any features

        # 2. Register new/modified events
        # ------------------------
        if not modifiedEvent: return 
        # modifies event bookkeeping
        if eventName not in self.newRecords:
            self.newRecords[eventName] = modifiedCategory
        if deleteOld != None:
            self.deletedRecords.add(deleteOld)

# ---------------------------------------

    def _LogCategorical(self, patientID, base, category):
        '''
        Track categorical events and their possible category values (observed in
        the training data).

        '''
        if self.mode != "train":
            return
        if base not in self.categoricalEventRoots:
            self.categoricalEventRoots[base] = []
        self.categoricalEventRoots[base].append(category)

    def _LogContinuous(self, event, eventVal):
        if self.mode != "train":
            return
        if event not in self.continuousEventVals:
            self.continuousEventVals[event] = []
        self.continuousEventVals[event].append(eventVal)

    def _LogIndicator(self, indicator, indicatorVal):
        if self.mode != "train":
            return
        if indicator not in self.indicatorEventVals:
            self.indicatorEventVals[indicator] = []
        self.indicatorEventVals[indicator].append(indicatorVal)

    def _AggregateObservations(self, discretized, onehot=True, zerofilling=True, continuousNbins=5):
        '''
        As we are flattening time, for one patient, multiple observations
        could exist under a single event(feature). This function transform existing
        features to ensure all features have single, numeric values. 

        :params: discretized: if True, continuous events are encoded using histograms
        :params: onehot: if True, one reference category is dropped for all one-hot-encoded columns
        :params: zerofilling: if True, missing entries are imputed as zeros, otherwise imputed using
                              SoftImpute from fancy impute
        :params: continuousNbins: number of bins for fitting continuous features into
                                  histograms. Only used when discretized == True

        '''
        self.discretized = discretized
        # Categorical & indicator events are encoded the same way for discrete/mixture
        # Both types of events are essentially treated as categorical
        #   - One-hot encode
        #   - Remove a reference column if it's a "one-time" record
        #   - Don't remove a reference column if it's a "multiple-times" record
        #   - Have a presence_flag column: first as sum, later as indicator

        # Categorical:
        # -----------
        # self.categoricalEventRoots: 
        #   Before: List of all possible categorical values, with duplicates
        #   After: A tuple of (referenceCategory, list of possible categories w/o dup)
        for categoricalEvent in self.categoricalEventRoots:
            sortedAllCategories = list(set(self.categoricalEventRoots[categoricalEvent]))
            sortedAllCategories.sort()
            referenceCategory = sortedAllCategories[0]
            self.categoricalEventRoots[categoricalEvent] = \
                                         (referenceCategory, sortedAllCategories)
            # so the reference category is the first category by alphabetical order

        # Indicator:
        # ----------
        # self.indicatorEventVals:
        #   Before: List of possible indicator values, with duplicates
        #   After: A tuple of (referenceCategory, list of possible categories w/o dup)
        for indicatorEvent in self.indicatorEventVals:
            sortedAllValues = list(set(self.indicatorEventVals[indicatorEvent]))
            sortedAllValues.sort()
            referenceValue = sortedAllValues[0]
            self.indicatorEventVals[indicatorEvent] = (referenceValue, sortedAllValues)

        # Continuous:
        # -----------
        # self.continuousEventVals:
        #   Before: List of possible continuous values, with duplicates
        #   After: A tuple of either: (histEdges, histMappings) (discretized)
        #                     or:     ()(mixture)
        if self.discretized in ["discretized_1", "discretized_2"]:
            for continuousEvent in self.continuousEventVals:
                # Compute histogram for all continuous features
                eventVals = self.continuousEventVals[continuousEvent]
                if len(set(eventVals)) < continuousNbins:
                    histEdges = sorted(set(eventVals))
                else:
                    histEdges = self._HistByQuantiles(eventVals)
                try:
                    histMapping = set(np.digitize(eventVals, bins = histEdges[:-1]))
                except:
                    print(list(histEdges), set(eventVals))
                    assert(False)

                self.continuousEventVals[continuousEvent] = (histEdges, histMapping)
            else:
                pass

        newPatientRecords = dict()
        newRecords = dict()
        deletedRecords = set()
        rareEventScreen = dict.fromkeys(list(self.categoricalEventRoots.keys()) + \
                                        list(self.continuousEventVals.keys()) + \
                                        list(self.indicatorEventVals.keys()), 0)
        eventLenAverage = dict.fromkeys(list(rareEventScreen.keys()), (0, 0))

        categoricalOneHotSetKnown = ["ethnicity", "gender", "admission_location", \
                                    "admission_type", "insurance", "marital_status", \
                                    "age_at_inicu"]

# <------------------------------------>
        # ########################
        # # Testing script below
        # # For visualizing features in R

        # featureTree = []

        # allUnderstoodEvents = list(self.categoricalEventRoots.keys()) + \
        #     list(self.continuousEventVals.keys()) + list(self.indicatorEventVals.keys())

        # for event in allUnderstoodEvents:
        #     nodes = [x.replace(",", ".") for x in event.split(":")]
            
        #     if event in self.categoricalEventRoots:
        #         currGroup = "categorical"
        #         leafNode = str(self.categoricalEventRoots[event][1])   # sorted categories
        #     elif event in self.continuousEventVals:
        #         currGroup = "continuous"
        #         leafNode = str(sorted(self.continuousEventVals[event]))     # sorted numbers
        #     elif event in self.indicatorEventVals:
        #         currGroup = "indicator"
        #         leafNode = str(self.indicatorEventVals[event][1]) # sorted indicator vals

        #     # root
        #     featureTree.append(["pancreatitis", nodes[0], "NA"])
        #     fromNode = nodes[0]
        #     toNode = nodes[0]
        #     for i in range(len(nodes) - 1):
        #         fromNode = ":".join(nodes[:i+1])
        #         toNode = ":".join(nodes[:i+2])
        #         featureTree.append([fromNode, toNode, "NA"])
        #     # leaf
        #     featureTree.append([toNode, leafNode.replace(",", "."), currGroup])

        # np.savetxt('panc_feature_tree.csv', np.array(featureTree), \
        #            delimiter = ",", header = "from, to, group", fmt = "%s")

        # print("Feature Tree Saved!")
        # sys.exit(0)

        # ########################
# <------------------------------------>

        sep = ":::"
        missingFlag = sep + "~missing_flag"

        for patientID in self.patientRecords:
            if patientID not in newPatientRecords:
                newPatientRecords[patientID] = dict()

            # 1. Categorical
            for categoricalEvent in self.categoricalEventRoots:

                if not self.discretized.startswith("discretized"):
                    newRecords[categoricalEvent + missingFlag] = 'indicator'
                # 1.1 Encode: per category by count, missing flag
                # 1.2 Track: rare events, average length of patient records

                if categoricalEvent in self.patientRecords[patientID]:
                    currCounter = collections.Counter(\
                                self.patientRecords[patientID][categoricalEvent])

                    if not self.discretized.startswith("discretized"):
                        newPatientRecords[patientID][categoricalEvent + missingFlag] = 0

                    # rare
                    if categoricalEvent in rareEventScreen:
                        rareEventScreen[categoricalEvent] += 1
                        if rareEventScreen[categoricalEvent]/self.totalNumPatients > self.rareFrequency:
                            del rareEventScreen[categoricalEvent]
                    # avg length
                    total, n = eventLenAverage[categoricalEvent]
                    eventLenAverage[categoricalEvent] = \
                        (total + len(self.patientRecords[patientID][categoricalEvent]), n+1)                

                else:
                    currCounter = collections.Counter([]) if zerofilling else None

                    if not self.discretized.startswith("discretized"):
                        newPatientRecords[patientID][categoricalEvent + missingFlag] = 1

                if type(currCounter) != type(None):
                    _, sortedAllCategories = self.categoricalEventRoots[categoricalEvent]
                    for category in sortedAllCategories:
                        categoricalEventFullName = categoricalEvent + sep + category
                        newPatientRecords[patientID][categoricalEventFullName] = \
                                                                currCounter[category]
                        newRecords[categoricalEventFullName] = 'categorical'

                    deletedRecords.add(categoricalEvent)

            # 2. Continuous
            for continuousEvent in self.continuousEventVals:

                if not self.discretized.startswith("discretized"):
                    newRecords[continuousEvent + missingFlag] = 'indicator'

                if continuousEvent in self.patientRecords[patientID]:

                    if not self.discretized.startswith("discretized"):
                        newPatientRecords[patientID][continuousEvent + missingFlag] = 0
                    # rare
                    if continuousEvent in rareEventScreen:
                        rareEventScreen[continuousEvent] += 1
                        if rareEventScreen[continuousEvent]/self.totalNumPatients > self.rareFrequency:
                            del rareEventScreen[continuousEvent]
                    # avg length
                    total, n = eventLenAverage[continuousEvent]
                    eventLenAverage[continuousEvent] = \
                        (total + len(self.patientRecords[patientID][continuousEvent]), n+1)

                else:
                    if not self.discretized.startswith("discretized"):
                        newPatientRecords[patientID][continuousEvent + missingFlag] = 1
                    if zerofilling:
                        self.patientRecords[patientID][continuousEvent] = []

                if continuousEvent in self.patientRecords[patientID]:
                    if discretized == "discretized_1":
                        histEdges, histMapping = self.continuousEventVals[continuousEvent]
                        histEventValPairs = self._FromHistToEvent(continuousEvent, \
                            histEdges, histMapping, self.patientRecords[patientID][continuousEvent])
                        for (histEvent, histVal) in histEventValPairs:
                            newRecords[histEvent] = 'continuous'
                            newPatientRecords[patientID][histEvent] = histVal
                    elif discretized == "discretized_2":
                        histEdges, histMapping = self.continuousEventVals[continuousEvent]
                        summaryEventValPairs = self._FromRecordsToSummary(\
                            continuousEvent, self.patientRecords[patientID][continuousEvent])
                        for (summaryEvent, summaryVal) in summaryEventValPairs:
                            newRecords[summaryEvent] = 'continuous'

                            if summaryEvent.endswith('~len'):
                                newPatientRecords[patientID][summaryEvent] = summaryVal
                            else:
                                summaryValMapped = np.digitize(summaryVal, histEdges[:-1])
                                summaryValMapped = max(min(histMapping), \
                                        min(summaryValMapped,max(histMapping)))
                                newPatientRecords[patientID][summaryEvent] = \
                                            summaryValMapped - 1 # starting from 0
                    else:   # non discretized
                        # encode min, max, median, most_recent, and len
                        summaryEventValPairs = self._FromRecordsToSummary(\
                            continuousEvent, self.patientRecords[patientID][continuousEvent])
                        for (summaryEvent, summaryVal) in summaryEventValPairs:
                            newRecords[summaryEvent] = 'continuous'
                            newPatientRecords[patientID][summaryEvent] = summaryVal

                deletedRecords.add(continuousEvent)

            # 3. Indicator
            for indicatorEvent in self.indicatorEventVals:
                _, sortedAllValues = self.indicatorEventVals[indicatorEvent]
                indicatorAsCategorical = True if len(sortedAllValues) != 1 else False
                # single-occurrence indicators do not have missing flags
                # since only the missing samples have their values as zeros

                if indicatorEvent in self.patientRecords[patientID]:
                    # handcrafted indicators
                    if "continuous_na" in indicatorEvent or \
                        "continuous_tr" in indicatorEvent:
                        continuousEvent = ":".join(indicatorEvent.split(":")[:-1])
                        # make sure to mark this patient as not missing this record
                        try:
                            if not self.discretized.startswith("discretized"):
                                assert((continuousEvent + missingFlag) in newRecords)
                                newPatientRecords[patientID][continuousEvent + missingFlag] = 0
                        except:
                            print("[WARNING] Found continuous events with all Nas", continuousEvent)

                    # elif "indicator_na" in indicatorEvent:
                    #     indicatorEvent_ = ":".join(indicatorEvent.split(":")[:-1]) + ":+/-"
                    #     if len(self.indicatorEventVals[indicatorEvent_][1]) != 1:
                    #         newPatientRecords[patientID][indicatorEvent_ + missingFlag] = 0
                    #         newRecords[indicatorEvent_ + missingFlag] = 'indicator'

                    elif indicatorAsCategorical:
                        if not self.discretized.startswith("discretized"):
                            newPatientRecords[patientID][indicatorEvent + missingFlag] = 0
                            newRecords[indicatorEvent + missingFlag] = 'indicator'

                    currCounter = collections.Counter(\
                                    self.patientRecords[patientID][indicatorEvent])

                    # rare
                    if indicatorEvent in rareEventScreen:
                        rareEventScreen[indicatorEvent] += 1
                        if rareEventScreen[indicatorEvent]/self.totalNumPatients > self.rareFrequency:
                            del rareEventScreen[indicatorEvent]

                else:
                    currCounter = collections.Counter([]) if zerofilling else None

                    if indicatorAsCategorical:
                        if not self.discretized.startswith("discretized"):
                            newPatientRecords[patientID][indicatorEvent + missingFlag] = 1

                if type(currCounter) != type(None):
                    _, sortedAllValues = self.indicatorEventVals[indicatorEvent]
                    for indicatorValue in sortedAllValues:
                        indicatorEventFullName = indicatorEvent + sep + str(indicatorValue)
                        newPatientRecords[patientID][indicatorEventFullName] = \
                                                            currCounter[indicatorValue]
                        newRecords[indicatorEventFullName] = 'indicator'
                    deletedRecords.add(indicatorEvent)

        self.patientRecords = pd.DataFrame.from_dict(newPatientRecords, orient='index')

        # Now, a couple of procedures remove rare/excessive/reference columns for one-hot-encodings

        # remove low frequency columns and all of the derived columns
        dropColumns = set()
        for eventName in self.patientRecords.columns:
            if eventName.split(sep)[0] in rareEventScreen.keys():
                dropColumns.add(eventName)
                if eventName in newRecords:
                    del newRecords[eventName]
                else:
                    deletedRecords.add(eventName)
                # print(eventName, \
                #         rareEventScreen[":".join(eventName.split(sep)[:-1])])

        # remove duplicate columns
        for eventName in eventLenAverage:   # essentially looping through all events
            total, n = eventLenAverage[eventName]
            if eventName in self.continuousEventVals and \
                                            self.discretized != "discretized_1":
                avg = total/n
                if avg < 5:
                    # only keep len and most recent
                    sep = ":::"
                    dropSuffixs = [sep+'~min', sep+'~max', sep+'~median']
                    if avg == 1:   # such as age
                        dropSuffixs.append(sep+'~len')
                else:
                    dropSuffixs = []

                for dropSuffix in dropSuffixs:
                    dropColumns.add(eventName + dropSuffix)
                    if (eventName + dropSuffix) in newRecords:
                        del newRecords[(eventName + dropSuffix)]

            # remove reference columns
            if onehot == 'reference' and \
                        eventName in self.continuousEventVals and \
                        self.discretized == 'discretized_1':
                avg = total/n
                if avg == 1 or eventName in categoricalOneHotSetKnown:
                    histEdges, histMapping = self.continuousEventVals[eventName]
                    histEventValPairs = self._FromHistToEvent(eventName, \
                        histEdges, histMapping, [histEdges[0]])
                    referenceEvent,_ = histEventValPairs[0]
                    dropColumns.add(referenceEvent)
                    if referenceEvent in newRecords:
                        del newRecords[referenceEvent]

            if onehot == 'reference' and eventName in self.categoricalEventRoots:
                avg = total/n
                if avg == 1 or eventName in categoricalOneHotSetKnown:
                    referenceCategory, _ = self.categoricalEventRoots[eventName]
                    dropColumns.add(eventName + sep + referenceCategory)
                    if (eventName + sep + referenceCategory) in newRecords:
                        del newRecords[eventName + sep + referenceCategory]

            if onehot == 'reference' and eventName in self.indicatorEventVals:
                referenceValue, sortedAllValues = self.indicatorEventVals[eventName]
                if len(sortedAllValues) != 1:
                    assert(len(sortedAllValues) == 2)
                    dropColumns.add(eventName + sep + str(referenceValue))
                    if (eventName + sep + str(referenceValue)) in newRecords:
                        del newRecords[eventName + sep + str(referenceValue)]
                        assert((eventName + sep + str(sortedAllValues[1])) in newRecords)
                        assert((eventName + missingFlag) in newRecords)

        self.patientRecords = self.patientRecords.drop(columns=list(dropColumns))

        if zerofilling:
            hasNan = np.sum(np.isnan(self.patientRecords.values), axis=0) != 0
            for NanEvent in self.patientRecords.columns[hasNan]:
                print("[WARNING] Nan events found even when zerofilling", NanEvent)

        self.newRecords = newRecords
        self.deletedRecords = deletedRecords

    def _FromHistToEvent(self, event, histEdges, histMapping, eventArray, sep=":::"):
        # histMapping contains mapped value of training event values: in [1, 3]

        eventArray = np.digitize(eventArray, histEdges[:-1])
        eventArray = [max(min(histMapping), min(x,max(histMapping))) for x in eventArray]
        # must be a subset of histMapping, exception for out of range test samples
        eventArrayCounts = collections.Counter(list(eventArray))

        eventValPairs = []

        for mappedVal in histMapping:
            lo = histEdges[mappedVal - 1]
            hi = histEdges[mappedVal]
            eventName = event + sep + "(BIN#%i):(%.2f-%.2f)" % (mappedVal, lo, hi)
            eventValPairs.append((eventName, eventArrayCounts[mappedVal]))

        return eventValPairs

    def _HistByQuantiles(self, vals, bins = 5):
        qs = np.linspace(0, 1, bins + 1)
        quantileEdges = list(np.quantile(vals, qs)) # note that sometimes due to floating point issues
        # this array might not be monotonic, correction as below
        for i in range(1, len(quantileEdges)):
            quantileEdges[i-1] = min(quantileEdges[i-1], quantileEdges[i])
            # the prior edge should be at most no larger than the next
        return quantileEdges
        # only use histEdges[:-1] for np.digitize

    def _FromRecordsToSummary(self, event, eventArray, sep = ':::'):
        # encode min, max, median, most_recent, and len
        statMin = min(eventArray) if eventArray != [] else 0
        statMax = max(eventArray) if eventArray != [] else 0
        statMedian = statistics.median(eventArray) if eventArray != [] else 0
        mostRecent = eventArray[-1] if eventArray != [] else 0
        eventLen = len(eventArray)

        eventSuffix = [sep+'~min', sep+'~max', sep+'~median', sep+'~most_recent', sep+'~len']
        eventNames = [event + suffix for suffix in eventSuffix]
        eventVals = [statMin, statMax, statMedian, mostRecent, eventLen]
        return zip(eventNames, eventVals)

    def ImputeMissing(self, zerofilling):
        '''
        Impute missing & remove columns that would cause problems during
        standardization.

        '''
        if self.verbose:
            print("[LOG]Impute missing value.")

        if zerofilling:
            # there should not be any missing values
            imputedDataframe = self.patientRecords
            x_test_df = imputedDataframe.loc[self.testIds]
            x_test_df_vals = x_test_df.values
            x_train_df = imputedDataframe.loc[self.trainIds]
            x_train_df_vals = x_train_df.values

        else:
            # Impute testing on both training and testing
            imputedResult = SoftImpute(verbose=False).fit_transform(\
                                                        self.patientRecords.values)
            imputedDataframe = pd.DataFrame(imputedResult, \
                                                columns=self.patientRecords.columns)
            imputedDataframe = imputedDataframe.set_index(\
                                                   self.patientRecords.index.values)
            x_test_df = imputedDataframe.loc[self.testIds]
            x_test_df_vals = x_test_df.values

            # Impute training data only using the training rows
            x_train_df = self.patientRecords.loc[self.trainIds]
            x_train_df_vals = SoftImpute(verbose=False).fit_transform(x_train_df.values)

        # Now the matrix is already "full", but standardization could numerically
        # cause issue due to low variance in some columns...

        # remove columns that would cause issues during standardization
        # only use training data to determine which columns should be removed
        train_mean = x_train_df_vals.mean(axis = 0)
        train_std = x_train_df_vals.std(axis = 0)
        x_train_standardized = (x_train_df_vals - train_mean) / train_std
        # only use training data to determine which columns should be removed
        # this ensures that feature engineering is not affected by new incoming testing data
        nan_free = np.sum(np.isnan(x_train_standardized), axis=0) == 0

        # if one event causes error, all relevant events are removed

        # for currFeature in imputedDataframe.columns:
        #     try:
        #         assert(":::" in currFeature)
        #     except:
        #         print("No separator detected for", currFeature)

        contains_nan = np.sum(np.isnan(x_train_standardized), axis=0) != 0
        contains_nan_features = imputedDataframe.columns[contains_nan]
        sep = ":::"
        contains_nan_features_roots = []
        for eventName in contains_nan_features:
            rootName = eventName.split(sep)[0]
            suffix = eventName.split(sep)[-1]
            if suffix not in ['~min', '~max', '~median', \
                              '~most_recent', '~len', '~missing_flag']:
                contains_nan_features_roots.append(rootName)
        # print(contains_nan_features)
        # print(contains_nan_features_roots)
        for allEvent in imputedDataframe.columns:
            contains_nan_features_roots_test = \
                sum([root + sep in allEvent for root in contains_nan_features_roots])
            if contains_nan_features_roots_test != 0:
                # print(allEvent)   # to see all the removed events
                nan_free[np.where(x_train_df.columns == allEvent)] = False

        x_train_df_ = pd.DataFrame(x_train_df_vals[:, nan_free], \
                                   columns = x_train_df.columns[nan_free])
        x_train_df = x_train_df_.set_index(x_train_df.index.values)

        x_test_df_ = pd.DataFrame(x_test_df_vals[:, nan_free], \
                                 columns = x_test_df.columns[nan_free])
        x_test_df = x_test_df_.set_index(x_test_df.index.values)

        self.imputedPatientRecords = pd.concat([x_train_df, x_test_df])
        self.imputedPatientRecords = self.imputedPatientRecords.astype("float64")

    def ExportData(self):
        '''
        Args: self.imputedPatientRecords

        Returns:
            trainDataframe, testDataframe, featurelist

        '''
        if type(self.imputedPatientRecords) == type(None):
            self.imputedPatientRecords = self.patientRecords

        # Load LOS and OUT info

        allPIDs, allLOSs, allTUDs = LoadLabels(self.SOURCES[self.sourceIndex],\
                                    labelPath = self.dataPath) # read txt
  
        PIDs = allPIDs[2]
        LOSs = allLOSs[2]
        # TUDs are ignored for this project

        PIDSubset = self.imputedPatientRecords.index
        LOSSubset = [LOSs[PIDs.index(pid)] for pid in PIDSubset]
        OUTSubset = [int(los != np.inf) for los in LOSSubset]
        self.imputedPatientRecords['LOS'] = LOSSubset
        self.imputedPatientRecords['OUT'] = OUTSubset

        # Export
        trainDataframe = self.imputedPatientRecords.loc[self.trainIds]
        testDataframe = self.imputedPatientRecords.loc[self.testIds]
        featurelist = self.imputedPatientRecords.drop(\
                                                columns=["LOS", "OUT"]).columns
        featurelist = np.array(featurelist)

        if self.verbose:
            print("[Feature] All data exported, total number of features:", len(featurelist))

        return trainDataframe, testDataframe, featurelist

def LoadLabels(dataset, labelPath):
    '''
    Returns labels including LOS, OUT, and TUD

    Args:
        dataset: str

    Returns:    
        PIDs: 3-tuple of trainPIDs, testPIDs, allPIDs,
              all of which are lists of ints
        LOSs: 3-tuple of lists of floats
        TUDs: 3-tuple of lists of floats

    -------
    Currently these are preprocessed and directly read from txt files.
    Different ways of computing LOS can be incorporated (future).

    '''
    labelPath = '{}{}{}'.format(labelPath, dataset, '_preprocessed_full')
    # labelPath = '{}{}{}'.format(labelPath, dataset, '_forecast_icu_los')
    # subsets = ['train', 'test']                                 # head
    subsets = ['full', 'full'] 
    fileNames = ['patients.txt', 'patient_ICU_LoS.txt', \
                'patient_time_until_death_from_in_ICU.txt']     # tail

    trainPIDs = LoadLabelsInt(labelPath,'{}_{}'.format(subsets[0], fileNames[0]))
    testPIDs = LoadLabelsInt(labelPath,'{}_{}'.format(subsets[1], fileNames[0]))
    PIDs = (trainPIDs, testPIDs, trainPIDs + testPIDs)

    trainLOSs = LoadLabelsFlt(labelPath,'{}_{}'.format(subsets[0], fileNames[1]))
    testLOSs = LoadLabelsFlt(labelPath,'{}_{}'.format(subsets[1], fileNames[1]))
    LOSs = (trainLOSs, testLOSs, trainLOSs + testLOSs) 

    trainTUDs = LoadLabelsFlt(labelPath,'{}_{}'.format(subsets[0], fileNames[2]))
    testTUDs = LoadLabelsFlt(labelPath,'{}_{}'.format(subsets[1], fileNames[2]))
    TUDs = (trainTUDs, testTUDs, trainTUDs + testTUDs) 

    return (PIDs, LOSs, TUDs)

def LoadLabelsInt(labelPath, filePath):
    fileName = os.path.join(labelPath, filePath)
    with open(fileName, 'r') as file:
        lines = file.readlines()
        PIDs = [int(float(line)) for line in lines]  
    return PIDs

def LoadLabelsFlt(labelPath, filePath):
    fileName = os.path.join(labelPath, filePath)
    with open(fileName, 'r') as file:
        lines = file.readlines()
        PIDs = [float(line) for line in lines]
    return PIDs










