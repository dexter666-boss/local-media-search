import unittest
from verification_core import grounded_decision,json_object


class VerificationTests(unittest.TestCase):
    def test_conditioned_name_claim_without_blind_evidence_is_not_accepted(self):
        raw=dict(verdict='match',evidence=['图上写着目标名称'],missing=[])
        result=grounded_decision(raw,dict(named_entities=['任意新地点']),
                                 dict(objects=['拱门和墙'],events=[],readable_text=['A区小道'],named_entities=[]))
        self.assertEqual(result['verdict'],'uncertain')
        self.assertNotIn('图上写着目标名称',result['evidence'])

    def test_name_can_be_supported_without_topic_specific_code(self):
        result=grounded_decision(dict(verdict='match',evidence=['标牌可读'],missing=[]),
                                 dict(named_entities=['New-Place']),
                                 dict(objects=['标牌'],events=[],readable_text=['new place'],named_entities=[]))
        self.assertEqual(result['verdict'],'match')

    def test_missing_condition_or_empty_evidence_cannot_confirm(self):
        for raw in [dict(verdict='match',evidence=[],missing=[]),dict(verdict='match',evidence=['猫'],missing=['数量'])]:
            self.assertEqual(grounded_decision(raw,dict(named_entities=[]),dict(objects=[],events=[]))['verdict'],'uncertain')

    def test_invalid_plans_and_answers_abstain(self):
        self.assertEqual(grounded_decision({}, {}, {})['verdict'],'uncertain')
        self.assertEqual(grounded_decision({}, dict(named_entities=[]), {})['verdict'],'uncertain')

    def test_fenced_json_and_broken_output(self):
        self.assertEqual(json_object('```json\n{"verdict":"reject"}\n```')['verdict'],'reject')
        self.assertEqual(json_object('not JSON'),{})


if __name__ == '__main__':
    unittest.main()
