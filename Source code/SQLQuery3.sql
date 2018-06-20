select q.*
from Posts q
join Posts a on a.Id = q.AcceptedAnswerId
where a.CreationDate between '2010-01-01' and '2010-01-31'
and q.AcceptedAnswerId IS NOT NULL
and q.PostTypeId = 1
and q.AnswerCount>=5