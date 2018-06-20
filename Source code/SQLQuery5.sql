select DISTINCT a.OwnerUserId
from Posts q
JOIN Posts a ON q.Id = a.ParentId
where q.CreationDate between '2010-01-01' and '2010-01-31'
and a.ParentId IS NOT NULL
and q.AnswerCount>=5
and q.PostTypeId=1
and q.AcceptedAnswerId IS NOT NULL