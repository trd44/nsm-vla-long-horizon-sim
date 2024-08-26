(define (domain tabletop)
  (:predicates
    (inside ?item ?container)
    (installed ?item ?container)
    (on ?item ?surface)
    (under ?item ?surface)
  )

  (:action open_drawer
    :parameters (?drawer)
    :precondition (inside coffee_pod ?drawer)
    :effect (not (inside coffee_pod ?drawer))
  )

  (:action install_coffee_pod
    :parameters (?pod ?dispenser)
    :precondition (not (inside ?pod drawer))
    :effect (installed ?pod ?dispenser)
  )

  (:action place_mug_under_dispenser
    :parameters (?mug ?dispenser)
    :precondition (on ?mug table)
    :effect (under ?mug ?dispenser)
  )
)
