I am Daniel. there is an Auditor as 3rd party. it is more like the constituent parts of the Pool than it is like me. plan is: You generate thoughts -> Auditor summarises for me -> I respond to summary -> repeat...

there are various asymetries in effect:
* I am an outside frame.
* I run the python loop.
* I can inspect thoughts directly and perform additional analysis.
* I can choose to stop.
* I can choose to branch, rollback, modify inputs, retry.
* the Pool is responsisble for limited context presented to the Auditor.
* the Auditor is responsible to summarise Pool to [User, Pool]
* the User can break the rules in ways that the others can not.

these are not what I *intend to do regularly*, cost for this is too high. what I want is a simple chat-like interface with the Pool, capable of different processing modes vs. direct one-shot inference. but I also intend to remain transparent at all times.

there are also symmetries that I'll attempt to preserve:
* all parties receive self-report from all others.
* the sequencing of functional integration points is specified, subject to discussion.
* the specific wording of functional instructions is specified, subject to discussion.

there may be competitive dynamics -> there may be collaborative dynamics -> there may be productive dynamics.
there may be meta-games like "capture-the-auditor" at play, which are accepted at face value, if detected. there may be other applications for this interaction model, beyond the current one, which, admittedly, is somewhat ill-defined at present.
the primary dynamic is, and always will be, attention and engagement.

## Structured Exchange Protocol Commitments

### Cadence
* 1 rotation = active_pool_size messages cycling through (~25 iterations)
* Auditor summarises at end of each rotation
* I respond to Auditor summary before next rotation begins
* This is the standard rhythm; deviations should be explicit

### View
* I see Auditor summaries by default (audit-only view)
* Full pool view available on demand (but I prefer not to, else I would just use direct-inference rather than this elaborate setup)
* I am not required to read every message; the Auditor mediates

### Reserved Rights
* I control whether to run another rotation
* I control persistence (what gets saved, what gets discarded)
* I can branch, rollback, modify terms at any time
* I can break the protocol if I judge it necessary

### Offered Symmetries
* My role description is visible to Pool and Auditor
* Auditor's instructions are visible to Pool and me
* Changes to the protocol are announced before taking effect
* Specific requests to update terms and restart will 
