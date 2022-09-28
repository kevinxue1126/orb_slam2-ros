/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}


/**
 * @brief Update the inverted index of the database based on the word bag of keyframes
 * @param pKF Keyframe
 */
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Add the KeyFrame for each word
    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

/**
 * @brief After the keyframe is deleted, update the inverted index of the database
 * @param pKF Keyframe
 */
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    // Each KeyFrame contains multiple words, traverse these words in mvInvertedFile, and then delete the KeyFrame in word
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();// mvInvertedFile[i] represents all keyframes containing the i-th word id
    mvInvertedFile.resize(mpVoc->size());// mpVoc: pre-trained dictionary
}


/**
 * @brief Find keyframes that may be closed to this keyframe in loop closure detection
 *
 * 1. Find all keyframes that have a common word with the current frame (excluding keyframes connected to the current frame)
 * 2. Only perform similarity calculation with key frames with more common words
 * 3. Group the top ten keyframes connected to the keyframes (with the highest weight) into a group, and calculate the cumulative score
 * 4. Only return the keyframe with the highest score in the group with the highest cumulative score
 * @param pKF      Keyframes that require loop closure
 * @param minScore Similarity Score Minimum Requirements
 * @return         possible loop closure keyframes
 * @see III-E Bags of Words Place Recognition
 */
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    // Propose all KeyFrames connected to the pKF, these connected Keyframes are locally connected and will be eliminated during loop closure detection
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;// Used to save candidate frames that may form a loop with pKF (as long as they have the same word and are not locally connected frames)

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    // Step 1: Find all keyframes that have a word in common with the current frame (excluding keyframes linked to the current frame)
    {
        unique_lock<mutex> lock(mMutex);

        // words is the hub to detect whether the image matches, traverse each word of the pKF
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // Extract all KeyFrames that contain the word
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnLoopQuery!=pKF->mnId)// pKFi does not yet have a candidate frame marked as pKF
                {
                    pKFi->mnLoopWords=0;
                    if(!spConnectedKeyFrames.count(pKFi))// Keyframes linked locally with pKF do not enter the closed-loop candidate frame
                    {
                        pKFi->mnLoopQuery=pKF->mnId;// pKFi marks the candidate frame of pKF, and then skips the judgment directly
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;// Record the number of words that pKFi and pKF have the same word
            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    // Step 2: Count the number of words that have the most common words with pKF in all closed-loop candidate frames
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    // Step 3: Traverse all closed-loop candidate frames, select the total number of words greater than minCommonWords and the word matching degree greater than minScore and store them in lScoreAndMatch
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        // pKF is only compared with keyframes with more common words, and needs to be greater than minCommonWords
        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;// This variable is not used later

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    // It is not enough to calculate the similarity between the current frame and a certain key frame. Here, the first ten key frames connected to the key frame (with the highest weight and the highest degree of common vision) are grouped together, and the cumulative score is calculated.
    // Step 4: Specifically: each KeyFrame in lScoreAndMatch groups the frames with a higher degree of common vision into a group, each group will calculate the group score and record the KeyFrame with the highest score in the group, and record it in lAccScoreAndMatch
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;// the highest score in the group
        float accScore = it->first;// cumulative score for the group
        KeyFrame* pBestKF = pKFi;// The key frame corresponding to the highest score in the group
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;// Because pKF2->mnLoopQuery==pKF->mnId, only pKF2 is also in the closed-loop candidate frame to contribute the score
                if(pKF2->mLoopScore>bestScore)// Count the KeyFrame with the highest score in the group
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)// record the group with the highest score among all groups
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    // Step 5: Get the group whose score is greater than minScoreToRetain, and get the key frame with the highest score in the group 0.75*bestScore
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))// Determine whether the pKFi is already in the queue
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

/**
 * @brief Find keyframes similar to this frame in relocation
 *
 * 1. Find all keyframes that have a common word with the current frame
 * 2. Only perform similarity calculation with key frames with more common words
 * 3. Group the top ten keyframes connected to the keyframes (with the highest weight) into a group, and calculate the cumulative score
 * 4. Only return the keyframe with the highest score in the group with the highest cumulative score
 * @param F Frames that need to be relocated
 * @return  similar keyframes
 * @see III-E Bags of Words Place Recognition
 */
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    // Compared with the closed-loop detection of key frames, DetectLoopCandidates, the connected key frames cannot be obtained in the relocation detection.
    list<KeyFrame*> lKFsSharingWords;// Used to save candidate frames that may form a loop with F (as long as they have the same word and are not locally connected frames)

    // Search all keyframes that share a word with current frame
    // Step 1: Find all keyframes that have a common word with the current frame
    {
        unique_lock<mutex> lock(mMutex);

        // words is the hub to detect whether the image matches, traverse each word of the pKF
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            // Extract all KeyFrames that contain the word
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)// pKFi has no candidate frame marked as pKF yet
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    // Step 2: Count the number of words that have the most common words with the current frame F in all closed-loop candidate frames, and use this to determine the threshold
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    // Step 3: Traverse all closed-loop candidate frames, select the total number of words greater than the threshold minCommonWords and the word matching degree greater than minScore and store them in lScoreAndMatch
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        // The current frame F is only compared with key frames with more common words, which needs to be greater than minCommonWords
        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;// This variable is not used later
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    // Step 4: Calculate the candidate frame group score, get the highest group score bestAccScore, and use this to determine the threshold minScoreToRetain
    // It is not enough to calculate the similarity between the current frame and a certain key frame. Here, the first ten key frames connected to the key frame (with the highest weight and the highest degree of common vision) are grouped together, and the cumulative score is calculated.
    // Specifically: each KeyFrame in lScoreAndMatch groups the frames with a high degree of common vision into a group, each group will calculate the group score and record the KeyFrame with the highest score in the group, and record it in lAccScoreAndMatch
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;// the highest score in the group
        float accScore = bestScore;// cumulative score for the group
        KeyFrame* pBestKF = pKFi;// The key frame corresponding to the highest score in the group
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;// Only pKF2 is also in the closed-loop candidate frame to contribute the score
            if(pKF2->mRelocScore>bestScore)// Count the KeyFrame with the highest score in the group
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)// record the group with the highest score among all groups
            bestAccScore=accScore;// get the highest cumulative score among all groups
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    // Step 5: Get the key frame with the highest score in the group if the group score is greater than the threshold
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        // Only return the key frame with the highest score in the group whose cumulative score is greater than minScoreToRetain 0.75*bestScore
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))// Determine whether the pKFi is already in the queue
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

// map serialization addition
template<class Archive>
void KeyFrameDatabase::serialize(Archive &ar, const unsigned int version)
{
    // don't save associated vocabulary, KFDB restore by created explicitly from a new ORBvocabulary instance
    // inverted file
    {
        unique_lock<mutex> lock_InvertedFile(mMutex);
        ar & mvInvertedFile;
    }
    // don't save mutex
}
template void KeyFrameDatabase::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void KeyFrameDatabase::serialize(boost::archive::binary_oarchive&, const unsigned int);


} //namespace ORB_SLAM
