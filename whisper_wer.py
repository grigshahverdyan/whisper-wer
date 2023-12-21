import whisper
import os
import jiwer

model = whisper.load_model("medium")
references = [
                ["Get in touch with our tech support and remember that you can always reach us at this number."],
                ["Does this bus really go to the more? It goes all the way there. Are you sure? I know it does."],
                ["Let me check on the status of your order. Thank you for your patience."],
                ["In that case, let me tell you about some alternatives. Anything that works will be great."],
                ["Maybe then you will figure out how to follow the rules. If you kick me out, your rules won't matter."],
                ["Are you saying that we failed to deliver on time? I am sorry, I don't understand you."],
                ["Our room rates recently went up. Is that okay with you, Mrs. Smith? That should be okay."],
                ["I am sorry if you are having trouble with our package. Just to clarify, are you experiencing a delay?"],
                ["It looks like your order shipped to the wrong address. I'm so sorry about the mix-up."],  
                ["A lot of people prefer to use a disc or four, king at whom, and a tablet for when they travel."]               
]
transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
for itr, i in enumerate(os.listdir()):
    if (i[-3:] == 'wav'):
        result = model.transcribe(i)
        wer = jiwer.wer(
                        references[itr],
                        result['text'],
                        truth_transform=transforms,
                        hypothesis_transform=transforms,
                    )
        print(i, '\n','reference: ', references[itr], '\nhypothesis:', result["text"])
        print(f"Word Error Rate (WER) :", wer, '\n\n')